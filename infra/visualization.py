import matplotlib.pyplot as plt
import seaborn as sns
import random as rdn
import numpy as np
import pandas as pd

'''
Variable Name List:
	- data
	- daily_ret_list

'''
FIGWEDTH = 15
FIGHEIGHT = 5
TITLESIZE = 14

def _plot_deco(func):
	def wrapper(self, *arg, **kwarg):
		plt.figure(figsize=(FIGWEDTH, FIGHEIGHT))
		func(self, *arg, **kwarg)
		plt.legend(loc=2)
		plt.show()
		plt.close()
	return wrapper

def plot_pnl(daily_cumret_list, stat, label_list=[], title='', plot_scale='log'):
	pnl_count = len(daily_cumret_list)
	if len(label_list) == 0:
		label_list = ['pnl'+str(k+1) for k in range(pnl_count)]
	fig, ax = plt.subplots()
	fig.set_size_inches(FIGWEDTH, FIGHEIGHT)
	fig.suptitle(title, fontsize=TITLESIZE*1.2, y=0.95)  # Set font size for the title
	for i in range(pnl_count):
		ax.plot(daily_cumret_list[i], label=label_list[i], zorder=2.5)
	
	table = ax.table(cellText=stat, loc='upper left', cellLoc='center', bbox=[0.075, 0.84, 0.24, 0.14], zorder=5)  
	table.auto_set_font_size(False)  # Disable automatic font size adjustment
	table.set_fontsize(12)  # Set font size for table content
	table.scale(1, 1.5)  # Scale table vertically to make it easier to read
    
	ax.grid(True, which='major', axis='both', linestyle='-')
	ax.grid(True, which='minor', axis='both', linestyle='--')
    
	ax.legend(loc='upper left', bbox_to_anchor=(0., 1.), fontsize=12)  # Adjust legend position
	
	ax.set_yscale(plot_scale)
	fig.tight_layout()
	plt.show()


@_plot_deco
def plot_filled_lines(line_list, label_list=[], title=''):
	line_count = len(line_list)
	plt.title(title, fontsize=TITLESIZE)
	if len(label_list) == 0:
		label_list = ['line'+str(k+1) for k in range(line_count)]
	for i in range(line_count):
		plt.fill_between(line_list[0].index, 0, line_list[i], label=label_list[i])

@_plot_deco
def plot_ts(ts_list, label_list, title='', colorlist=[]):
	ts_count = len(ts_list)
	plt.title(title, fontsize=TITLESIZE)
	if len(label_list) == 0:
		label_list = ['line'+str(k+1) for k in range(ts_count)]
	for i in range(ts_count):
		if len(colorlist) == 0:
			plt.plot(ts_list[i], label=label_list[i])
		else:
			plt.plot(ts_list[i], label=label_list[i], color=colorlist[i])

@_plot_deco
def plot_n_instance_of_data(data, sample=10):
	sampling_ids = [rdn.randint(1, data.shape[1]) for i in range(sample)]
	for id in sampling_ids:
		plt.plot(data[:, id], label=str(id))