# -*- coding: utf-8 -*-

import numpy as np

class ZNorm():
    '''
    This class always uses the following dimension layout for all interfacing:
        [BATCH x TEMPORAL x NUM_VARIATE] [BxTxC]
    
    {
        assertions are used at all places to check the dimension layout.
        make sure to squeeze/unsqueeze the missing/extra dimensions as needed.
    }
    '''
    
    def __init__(self, ):
        self.eps = 1e-6
        
    def normalize(self, values):
        '''
        "normalize values using z-score standardization"
        ------------------------------------------------
        expecting values of shape [BxTxC]
        
        returns norm_values of shape
            [BxTxC]
        returns parameters of shape
            {'mean': [BxC], 'std': [BxC]}
        '''
        assert(values.ndim == 3), 'Expected values to be of shape [BxTxC].'
        # move temporal dimensiom to 0th place
        values = np.moveaxis(values, 1, 0) # new dimensions [TxBxC]
        # get parameters
        parameters = self._extract_parameters(values, temporal_first=True)
        mean = parameters['mean']
        std = parameters['std']
        # normalize
        norm_values = (values - mean) / (std + self.eps)
        # restore position of temporal dimension
        norm_values = np.moveaxis(norm_values, 0, 1) # new dimensions [BxTxC]
        return norm_values, parameters
    
    def denormalize(self, norm_values, parameters):
        '''
        "denormalize values using z-score standardization"
        --------------------------------------------------
        expecting norm_values of shape [BxTxC]
        expecting parameters of shape
            {'mean': [BxC], 'std': [BxC]}
        
        returns values of shape
            [BxTxC]
        '''
        assert(norm_values.ndim == 3), 'Expected values to be of shape [BxTxC].'
        # move temporal dimensiom to 0th place
        norm_values = np.moveaxis(norm_values, 1, 0) # new dimensions [TxBxC]
        # get parameters
        mean = parameters['mean']
        std = parameters['std']
        assert(mean.ndim == 2), 'Expected means to be of shape [BxC].'
        assert(std.ndim == 2), 'Expected stds to be of shape [BxC].'
        # denormalize
        values = (norm_values * std) + mean
        # restore position of temporal dimension
        values = np.moveaxis(values, 0, 1) # new dimensions [BxTxC]
        return values
    
    def normalize_sample(self, values, parameters):
        '''
        "normalize values using z-score standardization
            where the mean & std are provided, not extracted from the values"
        ---------------------------------------------------------------------
        expecting values of shape [BxTxC]
        expecting parameters of shape
            {'mean': [BxC], 'std': [BxC]}
        
        returns norm_values of shape
            [BxTxC]
        '''

        assert(values.ndim == 3), 'Expected values to be of shape [BxTxC].'
        # move temporal dimensiom to 0th place
        values = np.moveaxis(values, 1, 0) # new dimensions [TxBxC]
        # get parameters
        mean = parameters['mean']
        std = parameters['std']
        assert(mean.ndim == 2), 'Expected means to be of shape [BxC].'
        assert(std.ndim == 2), 'Expected stds to be of shape [BxC].'
        # normalize
        norm_values = (values - mean) / (std + self.eps)
        # restore position of temporal dimension
        norm_values = np.moveaxis(norm_values, 0, 1) # new dimensions [BxTxC]
        return norm_values
    
    def _extract_parameters(self, values, temporal_first):
        '''
        get parameters for normalization of values
        expecting values of shape
            [timestamps x num_metrics]
        '''
        if not temporal_first:
            # move temporal dimensiom to 0th place
            assert(values.ndim == 3), 'Expected values to be of shape [BxTxC].'
            values = np.moveaxis(values, 1, 0) # new dimensions [TxBxC]
        else:
            # data is already in temporal first format
            pass
        # get parameters
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        parameters = {'mean': mean, 'std': std}
        return parameters
    
    def __repr__(self, ):
        return 'class for normalizing time-series using z-score \
            standardization.'

class MinMax():
    def __init__(self,):
        self.eps = 1e-6
    