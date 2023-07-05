# global package
import numpy as np
from numba import njit, prange
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

# alphahelix package
import op_20230620 as op

'''
Alpha Research Protocol:
'''

def signal(db):
    univ = db['univ_spx500']
    max_indicator = op.ts_avg_of_nmax(db['c2c_ret'], 20, 5)
    max_indicator_csr = op.cs_rank(max_indicator, univ)
    choose_signal = np.where(max_indicator_csr<0.1, 1.0, 0.0)
    norm_signal = choose_signal/op.cs_sum(choose_signal, univ)
    return norm_signal

if __name__ == '__main__':
	config_url = '../descriptor.txt'
	config_dict = datareader.read_descriptor(config_url)
	database = datareader.data_generator(config_dict)
	signal = signal(database)
	backtest_result = bt(signal, config_dict)
	print(backtest_result['fitness'])

