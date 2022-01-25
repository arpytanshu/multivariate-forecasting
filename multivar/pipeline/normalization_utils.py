#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:58:08 2021

@author: ansh
"""

import numpy as np

class ZNorm():
    '''
    This class normalizes windows of data using z-score standardization.
    
    This class is flexible with the position of temporal dimension.
    The temporal dimension can be at any position &
    needs to be specified when the object is initialized.

    '''
    
    def __init__(self, temporal_axis, eps=1e-6):
        self.eps = eps
        self.temporal_axis = temporal_axis
        
    def normalize(self, values):
        '''
        "normalize values using z-score standardization"
        ------------------------------------------------
        '''
        # get parameters
        parameters = self._extract_parameters(values)
        mean = parameters['mean']
        std = parameters['std']
        # normalize
        norm_values = (values - mean) / (std + self.eps)
        return norm_values, parameters
    
    def denormalize(self, norm_values, parameters):
        '''
        "denormalize values using z-score standardization"
        --------------------------------------------------
        '''
        # get parameters
        mean = parameters['mean']
        std = parameters['std']
        # denormalize
        values = (norm_values * std) + mean
        return values
    
    def normalize_sample(self, values, parameters):
        '''
        "normalize values using z-score standardization
            where the mean & std are provided, not extracted from the values"
        ---------------------------------------------------------------------
        '''
        # get parameters
        mean = parameters['mean']
        std = parameters['std']
        # normalize
        norm_values = (values - mean) / (std + self.eps)
        return norm_values
    
    def _extract_parameters(self, values):
        '''
        get parameters for normalization of values
        '''
        # get parameters
        mean = np.mean(values, axis=self.temporal_axis)
        mean = np.expand_dims(mean, axis=self.temporal_axis)
        
        std = np.std(values, axis=self.temporal_axis)
        std = np.expand_dims(std, axis=self.temporal_axis)

        parameters = {'mean': mean, 'std': std}
        return parameters
    
    def __repr__(self, ):
        return 'class for normalizing time-series using z-score \
            standardization.'

class MinMax():
    def __init__(self,):
        self.eps = 1e-6
    