import numpy as np
from numba import njit, prange
import pandas as pd
import bottleneck as bn
'''
圖像化report可跟易州對接
Output Name: indi 
'''
def cum_dailyret(daily_ret, compound=True):
	if compound:
		cumret = np.cumprod(1+daily_ret)
	else:
		cumret = 1+np.cumsum(daily_ret)
	return cumret

def ann_ret(daily_ret, periods_in_year=252, compound=True): #compound=False if Long-Short or other usage
	ts_length = len(daily_ret)
	cumret = cum_dailyret(daily_ret, compound)
	final_nav = cumret[-1]
	indi = final_nav ** (periods_in_year/ts_length) - 1
	return indi

def ann_std(daily_ret, periods_in_year=252):
	indi = np.std(daily_ret) * (periods_in_year ** 0.5)
	return indi

def ann_sharpe(daily_ret, periods_in_year=252, compound=True):
	ret = ann_ret(daily_ret, periods_in_year, compound)
	std = ann_std(daily_ret, periods_in_year)
	indi = ret/std
	return indi

def rolling_ret(daily_ret, rolling_period=252, compound=True):
	if compound:
		log_daily_ret = np.log(1+daily_ret)
		interval_ret = np.exp(bn.move_sum(log_daily_ret, rolling_period))
	else:
		interval_ret = bn.move_sum(daily_ret, rolling_period)+1
	indi = interval_ret ** (252/rolling_period) - 1
	return indi

def rolling_std(daily_ret, rolling_period=252):
	indi = bn.move_std(daily_ret, rolling_period) * (252**0.5)
	return indi

def rolling_sharpe(daily_ret, rolling_period=252, compound=True):
	rr = rolling_ret(daily_ret, rolling_period, compound)
	rs = rolling_std(daily_ret, rolling_period)
	indi = rr/rs
	indi[:rolling_period] = 0.0
	return indi

def _eachday_drawdown(cumret):
	indi = []
	historical_high = 1
	for t in range(len(cumret)):
		if historical_high > cumret[t]:
			indi.append(cumret[t] / historical_high - 1)
		else:
			historical_high = cumret[t]
			indi.append(0)
	indi = np.asarray(indi)
	return indi

def _drawdown_record(drawdown_record):
	dd_records = []
	current_dd_day_count, current_max_dd = 0, 0
	for t in range(len(drawdown_record)):
		current_dd = drawdown_record[t]
		if current_dd < 0:
			current_dd_day_count += 1
			if current_dd < current_max_dd:
				current_max_dd = current_dd
		elif current_dd == 0:
			if current_dd_day_count > 0:
				dd_records.append([current_dd_day_count, current_max_dd])
				current_dd_day_count, current_max_dd = 0, 0
	indi = np.asarray(dd_records)
	return indi

def max_n_drawdown(eachday_drawdown, n):
	dd_record = _drawdown_record(eachday_drawdown)
	dd_depth = dd_record[:, 1]
	sorted_dd = dd_depth[np.argsort(dd_depth, axis=0)]
	indi = np.mean(sorted_dd[:n])
	return indi

def _eachday_tvr(traded_signal, scale):
	total_trading = np.sum(traded_signal, axis=1)
	indi = total_trading/scale
	return indi

@njit
def _lorenz_area(sorted_series):
	n = len(sorted_series)
	if n==0:
		return 0
	else:
		cumsum_series = np.cumsum(sorted_series)
		lorenz_curve = (np.arange(1, n+1) / n) - (np.cumsum(sorted_series) / cumsum_series[-1])
		indi = 2*np.sum(lorenz_curve) / n
		return indi

@njit
def gini_coef(series):
	posi_series = np.sort(series[series>=0])
	nega_series = np.sort(np.fabs(series[series<0]))
	posi_ele_gini = _lorenz_area(posi_series)
	nega_ele_gini = _lorenz_area(nega_series)
	len_posi, len_nega = len(posi_series), len(nega_series)
	indi = (posi_ele_gini*len_posi+nega_ele_gini*len_nega)/(len_posi+len_nega)
	return indi

@njit
def _eachday_gini(input, univ):
	allt, alls = input.shape
	indi = []
	for t in prange(allt):
		valid_datapoint = np.asarray([input[t, k] for k in range(alls) if univ[t, k]==True])
		currentday_ginicoef = gini_coef(valid_datapoint)
		indi.append(currentday_ginicoef)
	return indi

def local_corr(daily_ret): #have to deal with timeframe issue
	return 1

def _series_correlation(series_x, series_y):
	return 1