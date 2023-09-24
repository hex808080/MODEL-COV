import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from pqdm.processes import pqdm
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.ndimage import binary_dilation
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

def set_default_dict(input_dict, default_dict):

	# Create a copy of default_dict
	result_dict = default_dict.copy()

	for key, value in input_dict.items():
		if key in result_dict:
			result_dict[key] = value

	return result_dict
	
def set_custom_palette(input_data):

	if isinstance(input_data, str):
		if input_data in plt.colormaps():
			return plt.get_cmap(input_data)
		else:
			print(f"Palette '{input_data}' not found. Using the current palette.")
			return plt.get_cmap()
	elif isinstance(input_data, list) and len(input_data) > 0 and len(input_data[0]) >= 3:
		try: 
			return matplotlib.colors.ListedColormap(input_data)
		except:
			print("Invalid input data. Using the current palette.")
			return plt.get_cmap()
	else:
		print("Invalid input data. Using the current palette.")
		return plt.get_cmap()
		
def set_bar_palette(barlist, features, groups, palette = None):
	
	if palette: palette = set_custom_palette(palette)
	else: palette = set_custom_palette('tab10')

	if groups:
		c_i = [-1]*(len(groups.keys()) + 1)
		legend = [-1]*(len(groups.keys()) + 1)
		for i in range(len(barlist)):
			k_i = 0
			for k in groups.keys():
				if any(item in '#' + features[i] + '#' for item in groups[k]):
					c_i[k_i] = i
					legend[k_i] = list(groups.keys())[k_i]
					barlist[i].set_color(palette(k_i))
				k_i += 1
			if not(i in c_i): 
				c_i[-1] = i
				legend[-1] = 'Other'
				barlist[i].set_color(palette(len(groups.keys())))

		c_i = [c for c in c_i if c != -1]
		legend = [l for l in legend if l != -1]	   
	else:
		for i in range(len(barlist)): barlist[i].set_color(palette(0))
		c_i, legend = [0], ['Importance']	

	return barlist, c_i, legend, palette
	
def random_forest_plot(X, y, results, config):

	indx = 			np.array(results['importances']['index'])
	importances = 		np.array(results['importances']['mean'])
	importances_std = 	np.array(results['importances']['std'])	

	default_config = {
		'N_bars': 		np.min([30, len(indx)]),
		'figsize':		[6, 6],
		'fontsize':		12,
		'groups':		False,
		'palette':		None
	}

	config_plt = set_default_dict(config['random_forest_plot_config'], default_config)
	config['random_forest_plot_config'] = config_plt
	plt.rcParams.update({'font.size': int(0.9*config_plt['fontsize'])})

	N = np.min([len(indx), config_plt['N_bars']])
	indx_ = indx[0:N]
	features_ = [X.columns[j] for j in indx_]
	importances_ = importances[0:N]
	importances_std_ = importances_std[0:N]

	labels = list(set(y))
	table_plt = X.iloc[:, indx_]
	table_plt['group'] = y
	table_plt = pd.melt(table_plt, id_vars='group', var_name='feature')

	f, ax = plt.subplots(2, 1, figsize = config_plt['figsize'], sharex = True)

	# BAR PLOT
	x50 = np.linspace(0,N,100)
	p50 = np.round(np.cumsum(importances_), 2)
	p50 = np.round(np.interp(x50, np.linspace(0,N,N), p50), 2)
	plt.rcParams["hatch.linewidth"] = 4
	rec = ax[0].fill_between([-1,x50[np.argmin(np.abs(p50-0.50))]],[1.05*np.max(importances_ + importances_std_)]*2, hatch='//', 
							 facecolor = 'tab:red', edgecolor = 'white', alpha = 0.3)
	
	barlist = ax[0].bar(range(N), importances_, yerr = importances_std_, 
		capsize = (ax[0].transData.transform((0.5, 0))[0] - ax[0].transData.transform((-0.5, 0))[0])/N/4*0.8, 
		error_kw = dict(lw = 1, ecolor = '#4B4B4B'))
	barlist, c_i, legend, palette = set_bar_palette(barlist, features_, config_plt['groups'], config_plt['palette'])

	ax[0].set_ylim(0,1.05*np.max(importances_ + importances_std_))
	ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax[0].tick_params(
		axis = 'x',		# changes apply to the x-axis
		which = 'both',		# both major and minor ticks are affected
		bottom = False,		# ticks along the bottom edge are off
		top = False,		# ticks along the top edge are off
		labelbottom = False)	# labels along the bottom edge are off
	ax[0].set_ylabel('Importance', fontsize = config_plt['fontsize'])
	ax[0].legend([barlist[i] for i in c_i] + [rec], legend + ['50% importance'])

	# BOXPLOT
	sns.boxplot(x = "feature", y = "value", hue = "group", data = table_plt, fliersize = 1,
		   palette = "Set2", dodge = True, linewidth = 1, ax = ax[1])
	plt.xticks(range(N), features_, rotation = 'vertical')

	ax[1].set_ylabel('Standardised data', fontsize = config_plt['fontsize'], labelpad = 12)

	handles, labels_ = ax[1].get_legend_handles_labels()
	plt.legend(handles, labels_, ncol = 2, loc = 'upper right')

	f.tight_layout(pad = 0, h_pad = 0, w_pad = 0)
	f.subplots_adjust(hspace = 0)

	title = labels[0] + ' vs ' + labels[1]
	ax[0].set_title(title, fontsize = config_plt['fontsize'], fontweight = 'bold')
	ax[1].set_xlabel('Features', fontsize = config_plt['fontsize'])
	
	plt.savefig(config['output'] + '.png', dpi = 300, bbox_inches = 'tight')
	return plt.gcf(), config
	
def random_forest_classify(X, y, config, plot_flag = True):

	config_rf = config['random_forest_config']
 
	n_splits, n_repeats = config_rf['N_split'], config_rf['N_iter']
	random_state = config_rf['random_state']

	scoring = {
		'F1': make_scorer(f1_score, average='macro'),
		'Precision': make_scorer(precision_score, average='macro'),
		'Recall': make_scorer(recall_score, average='macro')
		}
	if len(np.unique(y)) == 2:
		scoring['ROC-AUC'] = make_scorer(roc_auc_score, needs_proba = True)

	steps = []
	if config_rf['resampling'] == 'under': steps.append(('under', RandomUnderSampler(random_state = random_state)))
	if config_rf['resampling'] == 'SMOTE': steps.append(('over', SMOTE(random_state = random_state)))
	steps.append(('rf', RandomForestClassifier(n_estimators = config_rf['N_trees'], random_state = random_state)))

	clf = Pipeline(steps)

	cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = random_state)
	cv_results = cross_validate(clf, X, y, scoring = scoring, cv = cv, return_train_score = True, return_estimator = True, n_jobs = config['ncpu'])

	scores = np.array([cv_results['test_' + score] for score in scoring.keys()])
	scores_names = list(scoring.keys())
		
	importances = [cv_results['estimator'][i]['rf'].feature_importances_ for i in range(len(cv_results['estimator']))]
	importances_mean = np.mean(np.stack(importances), axis=0)
	importances_std = np.std(np.stack(importances), axis=0)	

	indx = np.argsort(importances_mean)[::-1]
	features = X.columns

	results = dict()
	results['scores'] = {'name': scores_names, 'mean': np.mean(scores, axis = 1).tolist(), 'std': np.std(scores, axis = 1).tolist()}
	results['importances'] = {
		'features': [features[j] for j in indx],
		'mean': importances_mean[indx].tolist(),
		'std': importances_std[indx].tolist(),
		'index': indx.tolist()
	}

	if plot_flag:
		img, config = random_forest_plot(X, y, results, config)
	
	return results

def random_forest(X, y, config = {}):

	if 'random_forest_config' in config.keys():
		config_rf = config['random_forest_config']
	else:
		config_rf = {}

	default_config = {
		'N_trees':	1000,
		'N_split':	10,
		'N_iter':	10,
		'N_shuffle':	100,
		'random_state':	42,
		'resampling':	'under'
	}

	config_rf = set_default_dict(config_rf, default_config)
	config['random_forest_config'] = config_rf
 
	results = random_forest_classify(X, y, config)
	
	results_rng = []
	t, d = [len(max(results['scores']['name'], key = len)) - len(t) for t in results['scores']['name']], [str(2 + abs(min(0, int(np.log10(abs(d)))))) for d in results['scores']['std']]
	if config_rf['N_shuffle'] and (config_rf['N_shuffle'] > 0):
		for n in tqdm (range(config_rf['N_shuffle']), desc = 'Randomising...'):
			y = y.sample(frac = 1, random_state = config_rf['random_state'])
			results_rng.append(random_forest_classify(X, y, config, None))

		scores_rng = np.array([results_rng[i]['scores']['mean'] for i in range(config_rf['N_shuffle'])])
		p = np.mean(scores_rng >= results['scores']['mean'], axis = 0)
		results['scores']['p'] = p.tolist()
		n = str(1 + int(np.ceil(np.log10(config_rf['N_shuffle']))))
		print('Scores:')
		for s_i in range(len(results['scores']['name'])):
			print(('\t' + ' '*t[s_i] + '%s:\t%.' + d[s_i] + 'f +- %.' + d[s_i] + 'f (p: %.' + n + 'f)') % (
				results['scores']['name'][s_i], 
				results['scores']['mean'][s_i], 
				results['scores']['std'][s_i], 
				results['scores']['p'][s_i]
			))
		
	else:
		print('Scores:')
		for s_i in range(len(results['scores']['name'])): 
			print(('\t' + ' '*t[s_i] + '%s:\t%.' + d[s_i] + 'f +- %.' + d[s_i] + 'f') % (
				results['scores']['name'][s_i], 
				results['scores']['mean'][s_i], 
				results['scores']['std'][s_i]
			))
	
	results_json = json.dumps(results, indent = 4)
	with open(config['output'] + '.json', 'w') as f: f.write(results_json)

	return results, results_rng

