# -*- coding: utf-8 -*-



import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/ansh/Documents/CHECKOUTS/multivar_tsfc')
from multivar.pipeline.train import stemgnn_wrapper
from multivar.pipeline.config import config

CHECKPOINT_BASE_PATH = '/tmp/output/'

# create checkpoint directories
# ------ ---------- -----------
result_file = os.path.join(CHECKPOINT_BASE_PATH, 'stemGNN', 'train')
result_test_file = os.path.join(CHECKPOINT_BASE_PATH, 'stemGNN', 'test')
if not os.path.exists(result_file): os.makedirs(result_file)
if not os.path.exists(result_test_file): os.makedirs(result_test_file)

num_timestamps_train = 8000
num_timestamps_test = 2000
num_metrics = 128

train_data = 100 * np.random.rand(num_timestamps_train*num_metrics).reshape(num_timestamps_train, num_metrics)
series_ts_train = np.arange(num_timestamps_train).reshape(-1, 1)
test_data = 100 * np.random.rand(num_timestamps_test*num_metrics).reshape(num_timestamps_test, num_metrics)
series_ts_valid = np.arange(num_timestamps_test).reshape(-1, 1)


something = stemgnn_wrapper(train_data = train_data,
                            valid_data = test_data,
                            series_ts_train = series_ts_train,
                            series_ts_valid = series_ts_valid,
                            config = config,
                            result_file = result_file)

something.train()

