import numpy as np
from numba import njit, prange
import pandas as pd
import datetime as dt
import os
import bottleneck as bn
import datareader

import op
import evaluation as ev
import visualization as vi

TICKSIZE = 1/750
BREADTH = 1/160

class Backtest():
	'''
	input: DataFrame / Array
	'''
	def __init__(self, signal, db, mode='research'): 
		self.o2o_ret = op.at_nan2zero(db['o2o_ret'])
		self.liquidity = 1e3 * op.ts_mean(db['volume']*db['close'], 21) #單位待確認
		self.dateindex, self.tickercolumn = db['date'], db['ticker']
		self.univ = db['univ']
		self.signal = signal
		self.scale = 1e6
		self.tax_rate = 5*1e-4 #10bps, spliting into half
		self.compound=True

		self.signal2position()
		self.position2retrun()
		if mode == 'research':
			self.advance_test()
			self.peres = self.evaluate(self.compound)

	def np2pdts(self, input):
		return pd.Series(input, self.dateindex)
		
	def signal2position(self):
		self.finite_signal = op.at_nan2zero(self.signal)
		self.normalized_signal = self.normalize_signal(self.finite_signal)
		self.scaled_signal = self.normalized_signal*self.scale
		self.traded_signal = np.fabs(op.ts_delta(self.scaled_signal, 1))
		self.traded_signal[0, :] = self.scaled_signal[0, :]

	def position2retrun(self):
		self.firm_grspnl = self.scaled_signal*self.o2o_ret
		self.market_impact = self.estimate_market_impact(self.traded_signal, self.liquidity)
		self.tax_fee = self.traded_signal*self.tax_rate
		self.firm_trancost = self.tax_fee+self.market_impact
		self.firm_netpnl = self.firm_grspnl - self.firm_trancost

		self.daily_grspnl = np.sum(self.firm_grspnl, axis=1)
		self.daily_trancost = np.nansum(self.firm_trancost, axis=1)
		self.daily_netpnl = self.daily_grspnl - self.daily_trancost

		self.daily_grsret = self.daily_grspnl / self.scale
		self.daily_netret = self.daily_netpnl / self.scale
		self.daily_grsret_pd = self.np2pdts(self.daily_grsret)
		self.daily_netret_pd = self.np2pdts(self.daily_netret)
		self.cumret_grs = ev.cum_dailyret(self.daily_grsret, self.compound)
		self.cumret_net = ev.cum_dailyret(self.daily_netret, self.compound)
		self.cumret_grsret_pd = self.np2pdts(self.cumret_grs)
		self.cumret_netret_pd = self.np2pdts(self.cumret_net)

	def advance_test(self):
		self.daily_tvr = ev._eachday_tvr(self.traded_signal, self.scale)
		self.daily_tvr_pd = self.np2pdts(self.daily_tvr)

		self.rolling_ir_pd = self.np2pdts(ev.rolling_sharpe(self.daily_netret, 252, self.compound))

		self.daily_retgini = ev._eachday_gini(self.firm_netpnl, self.univ)
		self.daily_posgini = ev._eachday_gini(self.normalized_signal, self.univ)
		self.daily_retgini_pd = self.np2pdts(self.daily_retgini)
		self.daily_posgini_pd = self.np2pdts(self.daily_posgini)
		
		self.daily_grsret_dd = ev._eachday_drawdown(self.cumret_grs)
		self.daily_grsret_dd_pd = self.np2pdts(self.daily_grsret_dd)
		self.daily_netret_dd = ev._eachday_drawdown(self.cumret_net)
		self.daily_netret_dd_pd = self.np2pdts(self.daily_netret_dd)


	def evaluate(self, compound=True):
		peres = {}
		peres['ret'] = ev.ann_ret(self.daily_netret, compound)
		peres['ir'] = ev.ann_sharpe(self.daily_netret, compound)
		peres['max_drawdown'] = ev.max_n_drawdown(self.daily_netret_dd, 1)
		peres['general_drawdown'] = ev.max_n_drawdown(self.daily_netret_dd, 5)
		peres['rdd'] = -peres['ret']/peres['max_drawdown']
		peres['tvr'] = np.nanmean(self.daily_tvr)
		posi_interval, nega_interval = np.sum(self.rolling_ir_pd>0), np.sum(self.rolling_ir_pd<0)
		peres['posi_ir_interval'] = posi_interval/(posi_interval+nega_interval)
		peres['ret_gini'] = np.nanmean(self.daily_retgini)
		peres['pos_gini'] = np.nanmean(self.daily_posgini)
		return peres

	def visualize(self, title=''):
		plot_scale = 'log' if self.compound==True else 'linear'
		backtest_stat = np.asarray([['Ret', 'IR', 'MDD', 'TVR'], 
			[str(np.round(100*self.peres['ret'], 2))+'%', 
			str(np.round(self.peres['ir'], 2)), 
			str(np.round(-100*self.peres['max_drawdown'], 2))+'%',
			str(np.round(-100*self.peres['tvr'], 2))+'%']])
		vi.plot_pnl([self.cumret_netret_pd, self.cumret_grsret_pd], backtest_stat, ['net', 'grs'], title+' PnL Graph', plot_scale)
		vi.plot_filled_lines([self.daily_netret_dd_pd, self.daily_grsret_dd_pd]
			, ['net', 'grs'], title+' DD Graph, max:'+str(np.round(-100*self.peres['max_drawdown'], 2))+'%')
		vi.plot_ts([self.rolling_ir_pd, pd.Series(np.linspace(0, 0, len(self.dateindex)), self.dateindex)]
			, ['net', 'zero-mark'], title+' Rolling IR, positive:'+str(np.round(100*self.peres['posi_ir_interval'], 2))+'%', ['steelblue', 'black'])
		vi.plot_ts([self.daily_retgini_pd, self.daily_posgini_pd]
			, ['gini of return: '+str(np.round(self.peres['ret_gini'], 2)), 
			'gini of weight: '+str(np.round(self.peres['pos_gini'], 2))], title+' Concentration Level')

	def normalize_signal(self, signal):
		signal_cs_sum = np.sum(np.fabs(signal), axis=1)
		signal_cs_sum_array = np.tile(signal_cs_sum, (signal.shape[1], 1)).T
		normalized_signal = signal/signal_cs_sum_array
		return normalized_signal
	
	def estimate_market_impact(self, traded_signal, past_amount): #估計式待確認
		tick_breadth = BREADTH*past_amount
		impacted_ticks = traded_signal/tick_breadth -1
		mi = np.where(impacted_ticks>0, 0.5*(impacted_ticks**2)*tick_breadth*TICKSIZE, 0.0)
		#mi = np.full_like(scaled_signal, 0.0)
		return mi
