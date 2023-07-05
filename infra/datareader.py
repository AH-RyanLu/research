import numpy as np
import pandas as pd
import datetime
from numba import prange, njit

#岳緯

def read_descriptor(txt_url):
	with open(txt_url, 'r') as file:
		raw_config = file.read()
	config_rows = raw_config.split('\n')
	config_dict = {x.split(':')[0]: x.split(':')[1].strip() for x in config_rows[1:]}
	config_dict['RequiredData'] = config_dict['RequiredData'].split(', ')
	config_dict['LongestDataRequired'] = int(config_dict['LongestDataRequired'])
	return config_dict

def univ_collector(univ_name):
	# go collecting 
	return list([1])

def get_data():
	return 1
