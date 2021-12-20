import numpy as np

class Config():
    def __init__(self, config_dict):
        self.__dict__ = config_dict
    def __repr__(self, ):
        return str(self.__dict__)

config_dict ={
    # dataloader stuff
    'train_length': 8,
    'valid_length': 2,
    'test_length': 0,
    
    'window_size': 128,
    'horizon': 8,

    # config for selecting series ix from multivariate data
    'num_series_in_data': 4,
    'num_target_series': 2,
    'get_target_series_ix': lambda: np.arange(config_dict['num_target_series']),
    # 'get_target_series_ix': lambda: return np.array([0,2]),
    
    # train stuff
    'optimizer': 'RMSProp',
    'lr': 0.0005,
    'batch_size': 64,
    'multi_layer': 5,
    'epoch': 50,
    'device': 'cuda',
    'decay_rate': 0.5,
    'dropout_rate': 0.5,
    'leakyrelu_rate': 0.2,
    'exponential_decay_step': 5,
    'early_stop': False,
    'early_stop_step': 2,
    'use_backcast_loss': True,


    'dataset': 'checkpoint',

    
    'train': True,
    'evaluate': True,
    'validate_freq': 1,
    'norm_method': 'z_score',
    'plot_attns': True,
    }


config = Config(config_dict)


