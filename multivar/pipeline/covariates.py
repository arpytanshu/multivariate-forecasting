# -*- coding: utf-8 -*-

import numpy as np
from pandas import to_datetime as to_dt

class CovGen():
    def __init__(self, cov_config):
        '''
        Class to generate temporal covariates from timestamps.
        Picks the required covariate's resolution and encoding type from 
            config.
        Supports 6 temporal resolutions:
            - month (1-12)
            - date of day (1-31)
            - weekday (0-6)
            - hour (0-23)
            - minute (0-59)
            - second (0-59)
        Supports 3 encoding method:
            - sinCos: encodes values b/w 0&1. Generates a Sin and Cosine
                encoded value for each covariate.
            - zeroOne: encodes values b/w 0&1, like minMAx scaling.
            - bool: all non zero values are 1, all zeros are 0.
        '''

        self.cov_config = cov_config
        self.max_vals = {
                # resolution: max_value
                'month':    12,
                'date':     31,
                'weekday':  7,
                'hour':     24,
                'minute':   60,
                'second':   60,
                }
        self.ts_extractor = {
            'month' :   lambda u_ts: np.array([x.month \
                                              for x in to_dt(u_ts, unit='s')]),
            'date':     lambda u_ts: (u_ts // 86400) % 31,
            'weekday':  lambda u_ts: (u_ts // 86400) % 7,
            'hour':     lambda u_ts: (u_ts // 3600) % 24,
            'minute':   lambda u_ts: (u_ts // 60) % 60,
            'second' :  lambda u_ts: (u_ts // 1) % 60,
            }
        
        self.encoder_fn = {
            # ['sinCos', 'zeroOne', 'bool']
            'sinCos': self._sinCos_encoder,
            'bool': self._bool_encoder,
            'zeroOne': self._zeroOne_encoder,
            }
        
        self.encoder_size = {
            'sinCos': 2,
            'bool': 1,
            'zeroOne': 1,
            None: 0,
            }
    
    def get_num_covs_from_config(self, ):
        num_covs = sum([self.encoder_size[x] for x in self.cov_config.values()])
        return num_covs
        
    def generate(self, unix_ts):
        temporal_covariates = []
        for resolution, enc_method in self.cov_config.items():
            if enc_method is not None:
                ts_extracted_feats = self.ts_extractor[resolution](unix_ts)
                encoded_vals = \
                    self.encoder_fn[enc_method](ts_extracted_feats,\
                                                self.max_vals[resolution])
                temporal_covariates.append(encoded_vals)
        temporal_covariates = np.hstack(temporal_covariates)
        return temporal_covariates
    
    def _sinCos_encoder(self, vals, max_num_val):
        cyclic = (vals * 2 * np.pi) / max_num_val
        sin_enc = np.sin(cyclic)
        cos_enc = np.cos(cyclic)
        return np.stack([sin_enc, cos_enc]).T
        
    def _bool_encoder(self, vals, max_num_val):
        return vals.astype(bool).astype(int).reshape(-1,1)
    
    def _zeroOne_encoder(self, vals, max_num_val):
        return (vals / vals.max()).reshape(-1,1)
    
    def __repr__(self,):
        return 'class to generate temporal covariates from timestamps.'
    
    
    


'''
+++++++++++++
+ interface +
+++++++++++++

import matplotlib.pyplot as plt
import numpy as np

cov_config = {
        # valid_encodings = ['sinCos', 'zeroOne', 'bool']
        # resolution: encoding
        'month':        'sinCos',
        'date':         'sinCos',
        'weekday':      None,
        'hour':         None,
        'minute':       None,
        'second':       None,
        }


CG = CovGen(cov_config)
self = CG

        
unix_ts = np.arange(100000000)[::1000]

covs = CG.generate(unix_ts)
print(covs.shape)


plt.figure(figsize=(20,5))
plt.plot(covs)

'''
