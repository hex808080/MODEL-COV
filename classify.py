import numpy as np
import pandas as pd
import statsmodels.api as sm
import json, argparse
from sklearn.impute import KNNImputer
import multiprocessing
import importlib
from datetime import datetime
import os

def categorical_to_int(data):
	# Convert categorical variables to integers
	data_col = data.columns[data.dtypes == 'object']

	for col in data_col:
		data[col] = pd.factorize(data[col])[0]
		
	return data
		
def clean_impute_standardise(data, labels, out = np.nan, thr = 1):

	# Set 0's to NaN
	data = data.mask(data == out, np.nan)

	# Remove full-NaN columns
	indx_col = np.sum(data.isnull(), axis=0) < data.shape[0] * thr
	data = data.loc[:, indx_col]
	
	# Remove full-NaN rows
	indx_row = (np.sum(data.isnull(), axis=1) < data.shape[1] * thr) & (~labels.isna())
	data = data.loc[indx_row, :]
	labels = labels.loc[indx_row]

	# Group-wise impute NaN values
	for g in pd.unique(labels):
		imp_mean = KNNImputer(missing_values=np.nan).fit(data.loc[labels == g,:])
		data.loc[labels == g,:] = imp_mean.transform(data.loc[labels == g,:])

	# Column-wise standardise data
	mu = np.mean(data, axis = 0)
	sd = np.std(data, axis = 0)
	for i in range(data.shape[1]):
		data.iloc[:,i] = (data.iloc[:,i] - mu[i])/sd[i]

	return data, labels

def regress_out(data, indep_name, ref_indx = None, sigLevel = None):

	dep_var = data.loc[:, ~data.columns.isin(indep_name)]
	indep_var = data.loc[:, data.columns.isin(indep_name)]

	if sigLevel == None: sigLevel = 0.05 #/len(dep_var.columns)
	else: sigLevel = float(sigLevel)

	hdr = list(indep_var.columns) + [cov + "_2" for cov in indep_var.columns]
	log = pd.DataFrame([], columns = [h + '_beta' for h in hdr] + ['const_beta'] + [h + '_pval' for h in hdr] + ['const_pval'], index = dep_var.columns)
	corr_var = dep_var.copy()
	for col in dep_var.columns:
		y, X_opt = dep_var[col], pd.concat([indep_var, indep_var**2], axis = 1)
		X_opt.columns = hdr
		X_opt = sm.add_constant(X_opt)
		model = sm.OLS(endog = y[ref_indx], exog = X_opt.loc[ref_indx,:], missing = 'drop').fit()
		pVals = model.pvalues
	
		while (pVals[np.argmax(pVals)] > sigLevel) and (len(pVals > sigLevel) > 1):
			X_opt = X_opt.drop(columns = X_opt.columns[np.argmax(pVals)])
			model = sm.OLS(endog = y[ref_indx], exog = X_opt.loc[ref_indx,:], missing = 'drop').fit()
			pVals = model.pvalues

		pVals_noConst = np.array([pVals[p_i] for p_i in range(len(pVals)) if not(pVals.keys()[p_i] == 'const')])
		if len(pVals_noConst):
			pred = model.predict(X_opt[list(pVals.keys())])
			residuals = y - pred
			if ('const' in pVals.keys()) and (pVals['const'] < sigLevel):
				residuals = residuals + pVals['const']

			corr_var[col] = residuals

		log.loc[col, [p + '_pval' for p in pVals.keys()]] = pVals.values
		log.loc[col, [p + '_beta' for p in pVals.keys()]] = model.params.values

	return corr_var, log

def make_groups(groups_list):
	new_list = []
	for items in groups_list:
		groups = items.split('+')
		new_list.append(groups)

	return(new_list)

def preprocess_data(table, labels, args):
	# Filter table
	if args.keep:
		if args.covariates: 	table = table[pd.unique(args.keep + [labels] + args.covariates)]
		else: 			table = table[pd.unique(args.keep + [labels])]
	elif args.exclude: 		table = table.drop(columns = args.exclude)

	if args.intermediates: table.to_csv(os.path.join(args.intermediates, 'table_filt.csv'), index = False)
	
	X, y = table.loc[:, ~(table.columns == labels)], table[labels]
	# Convert categorical independent variables (e.g. Gender: "M" / "F") to integers
	X = categorical_to_int(X)

	# Regress out covariates (if any)
	log = None
	if args.covariates:
		if args.reference:	ref_indx = y.isin(args.reference)
		else:			ref_indx = np.ones(y.size, dtype = bool) 
		X, log = regress_out(X, args.covariates, ref_indx, args.significance)

	if args.intermediates:
		table = pd.concat((X, y), axis = 1)
		table.to_csv(os.path.join(args.intermediates, 'table_filt_corr.csv'), index = False)
		if log is not None: log.to_csv(os.path.join(args.intermediates, 'table_filt_corr_beta.csv'))

	# Remove outliers, impute missing values and standardise dataset
	X, y = clean_impute_standardise(X, y)

	if args.intermediates: X.to_csv(os.path.join(args.intermediates, 'table_filt_corr_impstd.csv'), index = False)

	# Check groups
	if args.groups:
		groups = make_groups(args.groups)
		group_labels = [', '.join(g) if len(g) > 1 else g[0] for g in groups]
		y = y.astype(str)
		for g_i in range(len(groups)): 	y[y.isin(groups[g_i])] = group_labels[g_i]
		groups = group_labels
		X, y = X.loc[y.isin(groups), :], y[y.isin(groups)]

		table = pd.concat((X, y), axis = 1)
		table.sort_values(by = labels, key = lambda column: column.map(lambda e: group_labels.index(e)), inplace = True)
	else:	
		table = pd.concat((X, y), axis = 1)
		table = table.sort_values(by = labels)	

	if args.intermediates: table.to_csv(os.path.join(args.intermediates, 'table_filt_corr_impstd_grp.csv'), index = False)
	
	return table

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Process CSV file with optional inputs")
	parser.add_argument("filename",			type=str,					help="Input CSV filename.")
	parser.add_argument("labels",			type=str,					help="Name of the column containing group labels.")
	parser.add_argument("-g",	"--groups",	type=str, 	nargs='+', 	default=None,	help="Names of the groups to classify (default: use all groups).")		
	parser.add_argument("-c",	"--covariates",	type=str, 	nargs='+', 	default=None,	help="Name of the columns containing covariates to be regressed out from the dataset (default: no regression).")	
	parser.add_argument("-s",	"--significance", type=str, 			default=None,	help="Significance threshold for covariate regression, i.e. correction is performed if regression coefficient has p-value less than the threshold (default: 0.05).")
	parser.add_argument("-r",	"--reference",	type=str,	nargs='+',	default=None,	help="Reference group(s) on which to calculate regression coefficients. Only used if covariates are provided (default: regress over the whole dataset).")
	parser.add_argument("-e",	"--exclude",	type=str, 	nargs='+', 	default=None,	help="Name of the columns to exclude from the dataset (default: None).")
	parser.add_argument("-k",	"--keep",	type=str, 	nargs='+', 	default=None,	help="Name of the columns to keep from the dataset (default: None).")	
	parser.add_argument("-a",	"--algorithm",	type=str,	default="random_forest",	help="Name of algorithm to use. A python function with the same name must be available for import and contain a run(y, X, config) method (default: random_forest).")
	parser.add_argument("-j",	"--json_config",type=str,			default=None,	help="Name of JSON file containing additional parameters for classification and formatting (default: None).")
	parser.add_argument("-i",	"--intermediates", type=str,			default=None,	help="Name of the folder storing intermediate files, e.g. corrected table (default: None).")
	parser.add_argument("-o",	"--output",	type=str,			default=None,	help="Name prefix of output files (default: results_YYYYMMDD).")
	parser.add_argument("-n",       "--ncpu",       default=multiprocessing.cpu_count(),    	help="Number of CPUs to use (default: " + str(multiprocessing.cpu_count()) + ")")
	
	args = parser.parse_args()
	filename = args.filename
	labels = args.labels
	json_filename = args.json_config

	# Get input table
	if filename.endswith(".csv"):
		# Read CSV file into a DataFrame
		table = pd.read_csv(filename)
	elif filename.endswith((".xls", ".xlsx")):
		# Read Excel file into a DataFrame
		table = pd.read_excel(filename)
	else:
		print("ERROR: Unsupported file format. Please provide a CSV or Excel file.")
		exit()		
	
	# Check labels column exists within table
	if not(labels in table.columns):
		print("ERROR: Labels column" + labels + "does not exist in table " + filename + ".")
		exit()
	
	# Check config file. Create default config file if none is provided
	if json_filename:
		with open(json_filename) as config_json:
			config = json.load(config_json)
	else:
		config = {}
	
	# Check algorithm
	if args.algorithm:
		try: 			algorithm = importlib.import_module(args.algorithm)
		except ImportError:	print("ERROR: The specified module '{}' could not be imported.".format(args.algorithm))
		
	# Check output parameter
	if args.output:
		output = args.output
	else:
		today = datetime.now().strftime("%Y%m%d")
		output = 'results_' + today
	config['output'] = output

	# Create intermediates files folder (if any)
	config['intermediates'] = args.intermediates
	if config['intermediates'] and not(os.path.exists(config['intermediates'])): os.mkdir(config['intermediates'])

	# Check NCPU parameter
	config['ncpu'] = min(multiprocessing.cpu_count(), int(args.ncpu))

	# Perform filtering, covariate regression, imputation, standardisation 
	table = preprocess_data(table, labels, args)
	X, y = table.loc[:, ~(table.columns == labels)], table[labels]
	
	# Apply specified algorithm
	results = algorithm.run(X, y, config)

