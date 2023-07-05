import numpy as np
import pandas as pd
from numba import njit, prange
import bottleneck as bn

EPSILON = 1e-15

'''
Format:
Function Name List:
1. at: single-point operator
2. cs: cross-section operator: will require "univ"(necessary), "group_code" (conditional)
3. ts: time-series operator: if any 'NaN' exists in formation period, 'NaN' will be the output.
4. sup: other supporting functions like broadcasting series into array.
5. hidden layer within a big function should be denoted as _funtion_name()...

Function Input: 
case1~4:	nd.array(float64) -> nd.array(float64)
case5(sup): situational -> nd.array(float64)

Variable Name List:
	- input, input_x, input_y
	- univ
	- group
	- weight
	- days

- Make sure "one function does one job" to make function pools clear and easy to mainteinance.
- Patch note will be recorded afterward.

'''

# at - section #
def at_array2refdf(input, df):
	res = pd.DataFrame(input, df.index, df.columns)
	return res

def at_array2df(input, index, columns):
	res = pd.DataFrame(input, index, columns)
	return res

def at_df2array(input):
	return input.values

def at_sigmoid(input):
	return 1/(1+np.exp(-input))

def at_nan2zero(input):
	non_nanarray = np.where(input==input, input, 0)
	return non_nanarray

# cs - section #
@njit
def cs_rank(input, univ):
	res = np.copy(input)
	for di in prange(input.shape[0]):
		tmp = res[di, :]
		valid = np.isfinite(tmp) * univ[di]
		tmp[valid], group_min, group_max = _rankdata(tmp[valid])
		if group_max != group_min:
			tmp[valid] = (tmp[valid]-group_min) / (group_max-group_min)
		else:
			tmp[valid] = 0.5
		tmp[~valid] = np.nan
		res[di, :] = tmp
	return res

@njit
def _rankdata(cs):
	n = len(cs)
	ivec = np.argsort(cs)
	svec = cs[ivec]
	sumranks = 0
	dupcount = 0
	ranks = np.zeros(n, np.float64)
	for i in range(n):
		sumranks += i
		dupcount += 1
		if i == n-1 or svec[i] != svec[i+1]:
			averank = sumranks / float(dupcount) + 1
			for j in range(i-dupcount+1, i+1):
				ranks[ivec[j]] = averank
				if j == 0:
					minimum = averank
				elif j == n-1:
					maximum = averank
			sumranks = 0
			dupcount = 0
	return ranks, minimum, maximum

@njit
def cs_mean(input, univ):
	res = input.copy()
	valid = np.bitwise_and(np.isfinite(input), univ)
	for t in prange(input.shape[0]):
		cs = res[t, :]
		v_cs = valid[t, :]
		m = np.mean(cs[v_cs]) if len(cs[v_cs])>0 else np.nan
		cs[v_cs] = m
		cs[~v_cs] = np.nan
		res[t, :] = cs
	return res

@njit
def cs_sum(input, univ):
	res = input.copy()
	valid = np.bitwise_and(np.isfinite(input), univ)
	for t in prange(input.shape[0]):
		cs = res[t, :]
		v_cs = valid[t, :]
		m = np.sum(cs[v_cs]) if len(cs[v_cs])>0 else np.nan
		cs[v_cs] = m
		cs[~v_cs] = np.nan
		res[t, :] = cs
	return res

@njit
def cs_std(input, univ):
	res = input.copy()
	valid = np.bitwise_and(np.isfinite(input), univ)
	for t in prange(input.shape[0]):
		cs = res[t, :]
		v_cs = valid[t, :]
		s = np.std(cs[v_cs]) if len(cs[v_cs])>0 else np.nan
		cs[v_cs] = s
		cs[~v_cs] = np.nan
		cs[cs<EPSILON] = np.nan
		res[t, :] = cs
	return res

@njit
def cs_zscore(input, univ):
	m = cs_mean(input, univ)
	s = cs_std(input, univ)
	res = (input-m)/s
	return res

def cs_remove_absolute_outlier(input, pct, univ):
	res = input.copy()
	alls = input.shape[1]
	valid = np.isfinite(input) & univ
	csr = cs_rank(input, univ)
	ub, lb = 1.0 - pct/2, pct/2
	middle = np.bitwise_and(csr>=lb, csr<=ub)
	res[~middle] = np.nan
	res[~valid] = np.nan
	return res

def cs_remove_relative_outlier(input, pct, univ, nstd=3.5):
	res = input.copy()
	valid = np.isfinite(input) & univ
	input_remove_outlier = cs_remove_absolute_outlier(input, pct, univ)
	ub, lb = np.nanmax(input_remove_outlier, axis=1), np.nanmin(input_remove_outlier, axis=1)
	middle_std = np.nanstd(input_remove_outlier, axis=1)
	new_ub, new_lb = ub+nstd*middle_std, lb-nstd*middle_std
	new_ub_array, new_lb_array = sup_broadcast_1dimts(new_ub, input), sup_broadcast_1dimts(new_lb, input)
	new_middle = np.bitwise_and(input<=new_ub_array, input>=new_lb_array)
	res[~new_middle] = np.nan
	res[~valid] = np.nan
	return res

# cs_group -section #
def cs_grouprank(input, group, univ):
	return 1

# ts -section #

def ts_delay(input, days):
	res = np.pad(input,	((days, 0), (0, 0)), 'edge')[:-days]
	res[:days, :] = np.nan
	return res

def ts_delta(input, days):
	res = input - ts_delay(input, days)
	return res

def ts_sum(input, days):
	res = input.copy()
	res[np.isinf(res)] = np.nan
	for s in range(input.shape[1]):
		res[:, s] = bn.move_sum(res[:, s], days, min_count=1)
	res[np.fabs(res)<EPSILON] = np.nan
	res[~np.isfinite(input)] = np.nan
	return res

@njit
def _ts_weighted_moving_average(input, count, days, weight):
	allt, alls = input.shape
	res = np.full_like(input, np.nan)
	for s in prange(alls):
		for t in range(allt):
			tcut = min(t+1, days)
			summ = 0.0
			div = 0.0
			for tt in range(t, t-tcut, -1):
				delta = t - tt
				if count[tt, s] == 1:
					summ += weight[delta]*input[tt, s]
					div += weight[delta]*count[tt, s]
			if np.fabs(div) < EPSILON:
				res[t, s] = np.nan
			else:
				res[t, s] = summ / div
	return res

def ts_mean_linear(input, days):
	count = np.ones_like(input)
	count[~np.isfinite(input)] = 0.0
	weight = np.linspace(days, 1, days)
	res = _ts_weighted_moving_average(input, count, days, weight)
	res[~np.isfinite(input)] = np.nan
	return res

def ts_mean_exp(input, days, decay_rate):
	count = np.ones_like(input)
	count[~np.isfinite(input)] = 0.0
	weight = np.power(decay_rate, np.linspace(0, days-1, days))
	res = _ts_weighted_moving_average(input, count, days, weight)
	res[~np.isfinite(input)] = np.nan
	return res

def ts_mean(input, days):
	res = input.copy()
	res[np.isinf(res)] = np.nan
	for s in range(input.shape[1]):
		res[:, s] = bn.move_mean(res[:, s], days, min_count=1)
	res[~np.isfinite(input)] = np.nan
	return res

def ts_rank(input, days):
	unscaled_res = input.copy()
	unscaled_res[np.isinf(unscaled_res)] = np.nan
	for s in range(input.shape[1]):
		unscaled_res[:, s] = bn.move_rank(unscaled_res[:, s], days, min_count=2)
	res = unscaled_res/2.0 + 0.5
	res[~np.isfinite(input)] = np.nan
	return res

def ts_std(input, days):
	res = input.copy()
	res[np.isinf(res)] = np.nan
	for s in range(input.shape[1]):
		res[:, s] = bn.move_std(res[:, s], days, min_count=1)
	res[~np.isfinite(input)] = np.nan
	res[res<EPSILON] = np.nan
	return res

def ts_zscore(input, days):
	m = ts_mean(input, days)
	s = ts_std(input, days)
	res = (input-m)/s
	return res

# sup -section #

def sup_broadcast_1dimts(input, target_array):
	alls = target_array.shape[1]
	res = np.tile(input, (alls, 1)).T
	return res

def sup_broadcast_1dimcs(input, target_array):
	allt = target_array.shape[0]
	res = np.tile(input, (allt, 1))
	return res

# 1-dim series section #


