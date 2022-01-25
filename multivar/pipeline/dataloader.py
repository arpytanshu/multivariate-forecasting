#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:25:03 2021

@author: ansh
"""


import numpy as np
from .covariates import CovGen
from torch.utils.data import Dataset


class UltimateDataset(Dataset):
    def __init__(self, series_vals, series_ts, config):
        '''
        The utimate time-series dataloader for pytorch!
        The last dataloader you'll ever need for all pytorch time-series stuff!
        All numpy ops! no for loops! fast as fuck!
        
        - Handles window slicing and batching.
        - Handles covariates generation from config.
        - Handles full series & window normalizaton.
        - Handles data imputation/missing values.

        If normalization type is 'full':
            the parameters are saved as attributes
        If normalization type is 'window':
            the parameters are returned along with the batch.
        
        Parameters
        ----------
        series_vals : np.array
            numpy matric of shape [num_timestamps x num_metrics].
        series_ts : np.array
            numpy matrix of shape [num_timestamps x 1].
        config : Config
            configuration.
        '''
        self.config = config
        self.values = series_vals
        self.unix_timestamps = series_ts
        self.len = len(self.values)
        # self.batch = self.config.batch_size
        self.bc_len = self.config.bc_length
        self.fc_len = self.config.fc_length
        self.win_len = self.bc_len + self.fc_len
        self.num_window = self.len - self.win_len + 1
        
        self.Normalizer = self.config.normalization_method(temporal_axis=0)
        
        #if a window sampling logic is to be used, use it here!
        if self.config.training_sampler == 'random':
            self.sampled_indices = np.random.permutation(self.num_window)
        else: # use linear sampling, i.e. no shuffling.
            self.sampled_indices = np.arange(self.num_window)
        self.num_windows_sampled = len(self.sampled_indices)
        
        # gather normalization parameters
        # full_norm_params are used only if normalization type is 'full'.
        # params shape: R[1(B) x 1(T) x num_metrics]
        self.full_norm_params = \
            self.Normalizer._extract_parameters(self.values)

            
        # generate required covariates using unix_timestamps and config here.
        # generated covariates are of shape [num_timestamps x num_covariates]
        covGen = CovGen(self.config.covariate_config)
        self.num_covariates = covGen.get_num_covs_from_config()
        self.covariates = covGen.generate(self.unix_timestamps.ravel())
        
        
            
    def subseq2D(self, ts):
        # --------------------------------------------------------------------#
        # expected matrix of shape R[ num_timestamps x num_series ]
        # returns matrix of shape R[ num_samples x win_len x num_series ]
        # --------------------------------------------------------------------#
        shape = (ts.shape[0]-self.win_len+1, self.win_len, ts.shape[1])
        strides = (ts.strides[0], ts.strides[0], ts.strides[1])
        sliced = np.lib.stride_tricks.as_strided(ts,
                                                 shape=shape,
                                                 strides=strides)
        return sliced

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    def _get_sample(self, index):
        '''
        this batch will contain all windows generated from the indices
            [ start_ix : end_ix ]
        output dimensions: R[num_samples x num_timestamps x num_metrics]
        return None if batch index greater than length of dataset is queried.
        '''
        if index > self.__len__()-1:
            return None
        
        vals_start_ix = index
        vals_end_ix = index + self.win_len
        
        # window_ix = self.sampled_indices[start_ix:end_ix]
        # vals_start_ix = window_ix[0]
        # vals_end_ix = window_ix[-1] + self.win_len

        # make windows from series values
        # values_windows = \
            # self.subseq2D(self.values[vals_start_ix : vals_end_ix])
        # values_windows = values_windows[window_ix - window_ix[0]]
        values_windows = self.values[vals_start_ix : vals_end_ix]

        # make windows from covariates
        # covariates_windows = \
        #     self.subseq2D(self.covariates[vals_start_ix : vals_end_ix])
        # covariates_windows = covariates_windows[window_ix - window_ix[0]]
        covariates_windows = self.covariates[vals_start_ix : vals_end_ix]
        
        # slice into inputs and targets
        inp_vals = values_windows[:-self.fc_len, :] 
        tgt_vals = values_windows[-self.fc_len:, :]
        inp_covs = covariates_windows[:-self.fc_len, :] 
        tgt_covs = covariates_windows[-self.fc_len:, :]
        
        return inp_vals, tgt_vals, inp_covs, tgt_covs
    
    def __getitem__(self, index):
        # get raw batch
        '''
        inp_vals: R[temporal x num_variates]
        tgt_vals: R[temporal x num_variates]
        inp_covs: R[temporal x num_covariates]
        tgt_covs: R[temporal x num_covariates]
        '''
        inp_vals, tgt_vals, inp_covs, tgt_covs = self._get_sample(index)
        

        # NORMALIZATION
        # --------------------------------------------------------------------#
        # if normalization type is 'full':
        if self.config.normalization_type == 'full':
            inp_vals_norm = \
                self.Normalizer.normalize_sample(inp_vals, self.full_norm_params)
            tgt_vals_norm = \
                self.Normalizer.normalize_sample(tgt_vals, self.full_norm_params)
            norm_parameters = self.full_norm_params
                
        # if normalization type is 'window':
        elif self.config.normalization_type == 'window':
            inp_vals_norm, window_norm_params = \
                self.Normalizer.normalize(inp_vals)
            tgt_vals_norm = \
                self.Normalizer.normalize_sample(tgt_vals, window_norm_params)
            norm_parameters = window_norm_params
            
        # stitch covariates with values
        inputs = np.concatenate([inp_vals_norm, inp_covs], axis=1)
        targets = np.concatenate([tgt_vals_norm, tgt_covs], axis=1)
        return inputs, targets, norm_parameters
        
    def __len__(self):
        # tail = (1 if self.num_windows_sampled%self.batch>0 else 0)
        # return (self.num_windows_sampled // self.batch) + tail
        return self.num_window

'''
#%%

# ----------#
# interface #
# ----------#

from time import time
from torch.utils.data import DataLoader
from config import config
import numpy as np


# df = pd.read_csv('/home/ansh/Documents/GREENDECK_cliff/MULTIVARIATE_capacity_forecasting/**energy_dataset/ene_df_wo_fc.csv')
# series_vals = df.iloc[:, :7].values
# series_ts = 3600*np.arange(len(series_vals)) + 1588714200



num_timestamps = 50000
num_metrics = 128
series_vals = 100 * np.random.rand(num_timestamps*num_metrics).reshape(num_timestamps, num_metrics)
# series_vals = np.arange(num_timestamps*num_metrics).reshape(num_metrics, num_timestamps).T
series_ts = np.arange(num_timestamps).reshape(-1, 1)



config.normalization_type = 'full'
config.batch_size = 512
config.bc_length = 72
config.fc_length = 12

uds = UltimateDataset(series_vals, series_ts, config)
self = uds
uds_dl = DataLoader(uds, batch_size=config.batch_size, shuffle=True)

s = time()
for sample_ix in range(len(uds)):
    sample = uds[sample_ix]
e = time()
print('dataset.', e-s)
print(sample[0].shape)


s = time()
for _ in range(len(uds_dl)):
    batch = next(iter(uds_dl))
e = time()
print('dataloader.', e-s)
print(batch[0].shape)



#%%

config.normalization_type = 'full'
# config.batch_size = 64
config.bc_length = 72
config.fc_length = 12

uds = UltimateDataset(series_vals, series_ts, config)
# uds_dl = DataLoader(uds, batch_size=512, shuffle=True, num_workers=0)
self = uds
index = 0

s = time()
for batch_ix in range(len(uds_dl)):
    batch = next(iter(uds_dl))
e = time()
print('batching in dataloader.',  e-s)
print(batch[0].shape)


'''