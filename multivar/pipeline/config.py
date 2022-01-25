
from .normalization_utils import ZNorm, MinMax


    
function_mapper = {
    'normalization_method_zscore': ZNorm,
    'normalization_method_minmax': MinMax,
    }
    


class Config():
    def __init__(self, config_dict, function_mapper=None):
        self.__dict__ = config_dict        
        if function_mapper:
            # map functions to config objects:
            for key_string in function_mapper.keys():
                function = function_mapper[key_string]
                config_key = '_'.join(key_string.split('_')[:-1])
                config_value = key_string.split('_')[-1]            
            
                if self.__dict__[config_key] == config_value:
                    # print(config_key, function)
                    self.__dict__[config_key] = function
                
    def __repr__(self, ):
        return 'global configuration file.'



# VALUES ARE ONLY STRINGS
# ------ --- ---- -------
config_dict = { 

    # dataloader related config    
    'bc_length': 48,
    'fc_length': 12,
    'batch_size': 1,
    
    
    'covariate_config': {
        # valid_encodings = ['sinCos', 'zeroOne', 'bool']
        # resolution: encoding
        'month':        'sinCos',
        'date':         'sinCos',
        'weekday':      'sinCos',
        'hour':         'sinCos',
        'minute':       'sinCos',
        'second':       'sinCos',
        },
    
    
    # If training has to be performed on specifically sampled windows,
    # the index sampler method needs to be implemented, and its reference
    # be put into this configuration.
    # linear (default) / random
    'training_sampler': 'linear',
    # if True, stitches temporal covariates along the num_metric dimension.
    # if False, outputs only the series variates.
    'stitch_temporal_covariates': True,
    
    # NORMALIZATION
    # -------------
    # Opt for FULL normalization or WINDOW normalization!!
    # In full normalization, the entire metric data, 
    # both train and test segments are normalized using the same parameters.
    #
    # In window normalization, every input window and the corresponding target 
    # window is normalized individually.
    #
    # The normalized parameters returned by the dataloader in both these types
    # is different:
    # For full ts norm: return a mean/std pair per metric.
    # For window norm: return a mean/std pair per sample window.
    'normalization_type': 'full',
    'normalization_method': 'zscore', # MAPS TO FUNCTION CALL

    ## Model
    'device': 'cpu',
    'norm_method': 'z_score', #normalization method

    'optimizer': 'RMSProp', 
    'lr': 0.001, 
    'eps': 1e-08,
    'betas': (0.9, 0.999),
    'decay_rate': 0.0,

    #Training
    'epochs': 20,
    'use_backcast_loss': True,
    'plot_attns': False,
    'exponential_decay_step': 1,
    'validate_freq': 1,
    'early_stop': True,
    'early_stop_step': 10,

    ## STEM GNN
    # ----------
    'stemgnn_multi_layer': True, 
    'stemgnn_stack_cnt': 2, 
    'stemgnn_node_cnt': 128,
}



config = Config(config_dict, function_mapper)


