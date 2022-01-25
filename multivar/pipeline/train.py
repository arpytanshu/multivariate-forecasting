from ..stemGNN.base_model import Model
from ..stemGNN.forecast_dataloader import ForecastDataset, de_normalized
from ..stemGNN.eval_utils import evaluate
from .dataloader import UltimateDataset

import numpy as np
import os
import json
import torch
import time 

import torch.nn as nn
from datetime import datetime
import torch.utils.data as torch_data
import matplotlib.pyplot as plt



class stemgnn_wrapper:
    '''
    Wrapper of Stem GNN
    '''
    def __init__(self, train_data, valid_data, series_ts_train, series_ts_valid, config, result_file):
        '''
        Initialize the wrapper
        Input:
            train_data: training data
            valid_data: validation data
            config: configuration
            result_file: result file
        '''
        self.config = config
        self.node_cnt = self.config.stemgnn_node_cnt
        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        self.train_data = train_data
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')
        self.valid_data = valid_data
        self.series_ts_train = series_ts_train
        self.series_ts_valid = series_ts_valid
        self.result_file = result_file
    
    def build_model(self):
        self.model = Model(self.config.stemgnn_node_cnt, self.config.bc_length, self.config.stemgnn_multi_layer,
                           horizon=self.config.fc_length, stack_cnt=self.config.stemgnn_stack_cnt)
        self.model.to(self.config.device)
    
    def normalize(self):
        '''
        Normalize the data
        '''
        if self.config.norm_method == 'z_score':
            train_mean = np.mean(self.train_data, axis = 0)
            train_std = np.std(self.train_data, axis = 0)
            self.normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        elif self.config.norm_method == 'min_max':
            train_min = np.min(self.train_data, axis = 0)
            train_max = np.max(self.train_data, axis = 0)
            self.normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
        else:
            self.normalize_statistic = None
        if self.normalize_statistic is not None:
            with open(os.path.join(self.result_file, 'norm_stat.json'), 'w') as f:
                json.dump(self.normalize_statistic, f)
        
    def optimize_and_schedule(self):
        '''
        Optimize and schedule the learning rate, and the decay rate
        '''
        if self.config.optimizer == 'RMSProp':
            self.my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.config.lr, eps=self.config.eps)
        else:
            self.my_optim = torch.optim.Adam(params=self.model.parameters(),
                                        lr=self.config.lr, betas = self.config.betas)

        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.my_optim, gamma=self.config.decay_rate)
        
    def process_data(self):
        '''
        Process the data, load the dataset, and pass through dataloader
        '''
        self.train_set = UltimateDataset(self.train_data, self.series_ts_train, self.config)
        self.valid_set = UltimateDataset(self.valid_data, self.series_ts_valid, self.config)
        self.train_loader = torch_data.DataLoader(self.train_set, batch_size=self.config.batch_size, drop_last=False, shuffle=True,
                                            num_workers=0)
        self.valid_loader = torch_data.DataLoader(
            self.valid_set, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
    
    def init_forecast_loss(self):
        '''
        Initialize the forecast loss
        '''
        self.forecast_loss = nn.MSELoss(reduction='mean').to(self.config.device)
    
    def compute_total_trainable_params(self):
        '''
        Compute the total number of trainable parameters
        '''
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")
    
    def init_metrics(self):
        '''
        Initialize the metrics
        '''
        self.best_validate_mae = np.inf
        self.validate_score_non_decrease_count = 0
        self.performance_metrics = {}

    def save_model(self, model, model_dir, epoch=None):
        if model_dir is None:
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
        with open(file_name, 'wb') as f:
            torch.save(model, f)
    
    def load_model(self, model_dir, epoch=None):
        if not model_dir:
            return
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(file_name):
            return
        with open(file_name, 'rb') as f:
            model = torch.load(f)
        return model
    
    def inference(self, dataloader):
        forecast_set = []
        target_set = []
        backcast_set = []
        input_set = []
        normalizer = self.config.normalization_method(temporal_axis=0)
        self.num_covariates = self.train_set.num_covariates
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target, norm_parameters) in enumerate(dataloader):
                inputs = inputs.to(self.config.device).to(torch.float32)
                target = target.to(self.config.device).to(torch.float32)
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], self.config.fc_length,
                                          (self.node_cnt + self.num_covariates)], dtype=np.float)
                while step < self.config.fc_length:
                    forecast_result, a, backcast_result = self.model(inputs)
                    len_model_output = forecast_result.size()[1]
                    if len_model_output == 0:
                        raise Exception('Get blank inference result')
                    inputs[:, :self.config.bc_length - len_model_output, :] = inputs[:, len_model_output:self.config.bc_length,
                                                                        :].clone()
                    inputs[:, self.config.bc_length - len_model_output:, :] = forecast_result.clone()
                    forecast_steps[:, step:min(self.config.fc_length - step, len_model_output) + step, :] = \
                        forecast_result[:, :min(self.config.fc_length - step,
                                                len_model_output), :].detach().cpu().numpy()
                    step += min(self.config.fc_length - step, len_model_output)
                
                #remove covariates from the forecast steps
                forecast_steps = forecast_steps[:, :, :self.config.stemgnn_node_cnt]
                target = target[:, :, :self.config.stemgnn_node_cnt]
                #denormalize
                norm_parameters['mean'] = norm_parameters['mean'].detach().cpu().numpy()
                norm_parameters['std'] = norm_parameters['std'].detach().cpu().numpy()
                forecast_steps = normalizer.denormalize(forecast_steps, norm_parameters)
                target = normalizer.denormalize(target.detach().cpu().numpy(), norm_parameters)
                
                #concatenate the forecast and backcast
                forecast_set.append(forecast_steps)
                target_set.append(target)
                backcast_set.append(backcast_result.detach().cpu().numpy())
                input_set.append(inputs.detach().cpu().numpy())
        
        return np.concatenate(forecast_set, axis=0), \
            np.concatenate(target_set, axis=0), \
            np.concatenate(backcast_set, axis=0), \
            np.concatenate(input_set, axis=0)
    
    def validate(self, dataloader):
        start = datetime.now()
        forecast_norm, target_norm, _, _ = self.inference(dataloader)
        forecast, target = forecast_norm, target_norm
        score = evaluate(target, forecast)
        score_by_node = evaluate(target, forecast, by_node=True)
        end = datetime.now()

        score_norm = evaluate(target_norm, forecast_norm)
        print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
        print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
        if self.result_file:
            if not os.path.exists(self.result_file):
                os.makedirs(self.result_file)
            step_to_print = 0
            forcasting_2d = forecast[:, step_to_print, :]
            forcasting_2d_target = target[:, step_to_print, :]

            np.savetxt(f'{self.result_file}/target.csv', forcasting_2d_target, delimiter=",")
            np.savetxt(f'{self.result_file}/predict.csv', forcasting_2d, delimiter=",")
            np.savetxt(f'{self.result_file}/predict_abs_error.csv',
                    np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
            np.savetxt(f'{self.result_file}/predict_ape.csv',
                    np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

        return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                    rmse=score[2], rmse_node=score_by_node[2])

    def train(self):
        '''
        Iterates through epoch and save model
        '''
        # Pre-Requisites for training
        self.build_model()
        # self.normalize()
        self.optimize_and_schedule()
        self.process_data()
        self.init_forecast_loss()
        self.compute_total_trainable_params()
        self.init_metrics()
        #------------------------------------
        # Training
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.model.train()
            loss_total = 0
            cnt = 0
            attns = []
            for i, (inputs, target, norm_parameters) in enumerate(self.train_loader):
                inputs = inputs.to(self.config.device).to(torch.float32)
                target = target.to(self.config.device).to(torch.float32)
                self.model.zero_grad()
                forecast, attn, backcast = self.model(inputs)
                fc_loss = self.forecast_loss(forecast, target)
                if self.config.use_backcast_loss:
                    bc_loss = self.forecast_loss(backcast.squeeze(1).permute(0, 2, 1), inputs)
                    loss = fc_loss + bc_loss
                else:
                    loss = fc_loss
                cnt += 1
                loss.backward()
                self.my_optim.step()
                loss_total += float(loss)
                attns.append(attn.detach().cpu().numpy())
                if i % 10 == 0:
                    print('|', end='')

            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
            self.save_model(self.model, self.result_file, epoch)
            if self.config.plot_attns:
                attns = np.stack(attns).mean(axis=0)
                plt.imshow(attns)
                plt.show()

            if (epoch + 1) % self.config.exponential_decay_step == 0:
                self.my_lr_scheduler.step()
            if (epoch + 1) % self.config.validate_freq == 0:
                is_best_for_now = False
                print('------ validate on data: VALIDATE ------')
                self.performance_metrics = self.validate(self.valid_loader)
                if self.best_validate_mae > self.performance_metrics['mae']:
                    self.best_validate_mae = self.performance_metrics['mae']
                    is_best_for_now = True
                    self.validate_score_non_decrease_count = 0
                else:
                    self.validate_score_non_decrease_count += 1
                # save model
                if is_best_for_now:
                    self.save_model(self.model, self.result_file)
            # early stop
            if self.config.early_stop and self.validate_score_non_decrease_count >= self.config.early_stop_step:
                break
        return self.performance_metrics, self.normalize_statistic

    def test(self, test_data, series_ts_test, result_train_file):
        with open(os.path.join(result_train_file, 'norm_stat.json'), 'r') as f:
            normalize_statistic = json.load(f)
        self.model = self.load_model(result_train_file)
        self.node_cnt = test_data.shape[1]
        test_set = UltimateDataset(test_data, series_ts_test, self.config)
        test_loader = torch_data.DataLoader(test_set, batch_size=self.config.batch_size, drop_last=False,
                                            shuffle=False, num_workers=0)
        self.performance_metrics = self.validate(test_loader)
        mae, mape, rmse = self.performance_metrics['mae'], self.performance_metrics['mape'], self.performance_metrics['rmse']
        print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
