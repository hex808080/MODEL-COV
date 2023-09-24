import numpy as np
import pandas as pd
from glob import glob
import argparse
import os
import zipfile
import json
import subprocess
import nibabel as nib
from scipy.ndimage import binary_dilation
from datetime import datetime
from pqdm.processes import pqdm
from tqdm import tqdm 
import multiprocessing

def set_default_dict(input_dict, default_dict):

	# Create a copy of default_dict
	result_dict = default_dict.copy()

	for key, value in input_dict.items():
		if key in result_dict:
			result_dict[key] = value

	return result_dict

def check_config(config, conf):
	for key in config.keys():
		if type(config[key]) == str:
			config[key] = [config[key]]

	if (len(config['data']) == 0) or (len(config['rois']) == 0):
		print(config['data'], config['rois'])
		print('\nERROR: Configuration fields "data" and/or "rois" for', conf, 'do not exist or are empty lists. Please provide at least one element for both fields.')
		exit()
	else:
		if type(config['data']) == list:
			config['data_name'] = config['data']
		elif type(config['data']) == dict:
			config['data'], config['data_name'] = list(config['data'].values()), list(config['data'].keys())		
		
	if not((type(config['indx']) == int) or (type(config['indx']) == list) or (len(config['indx'])) or (type(config['indx']) == dict)):
		print('\nERROR: "indx" field for', conf, 'is undefined or unsupported. Please provide <int>, non-empty <list> or <dict>.')
		exit()
	else:
		if type(config['indx']) == int:
			config['indx'] = list(range(config['indx']))
		if type(config['indx']) == list:
			config['indx_name'] = [str(s) for s in config['indx']]
		elif type(config['indx']) == dict:
			config['indx'], config['indx_name'] = list(config['indx'].values()), list(config['indx'].keys())
		else:
			print('\nERROR. Unexpected "indx" value for', conf + '. Please check configuration file.')
			exit()
		
	if not(type(config['volumes'] == bool)):
		print('\nERROR: "volumes" field for', conf, 'must be <bool>. If specified, please set as <True> or <False>.')
		exit()
		
	return config
	
# Iterate over each subject and unzip relevant files into the corresponding folder
def unzip_subjects(folder):
	folder_path = os.path.join(directory, folder)
	zip_list = glob(folder_path + '/**/*.zip', recursive=True)
	res_list = set([res for feat in config.keys() for res in config[feat]['data'] + config[feat]['rois'] + config[feat]['lesions']])
	
	for res_file in res_list:
		for zip_file in zip_list:
			try:
				with zipfile.ZipFile(zip_file, 'r') as zip_ref:
					matching_files = [name for name in zip_ref.namelist() if (res_file in os.path.basename(name)) and ('.nii' in os.path.basename(name))]
					if matching_files:
						for matching_file in matching_files:
							# Get the filename without path
							filename = os.path.basename(matching_file)

							# Extract the file to the current working directory
							zip_ref.extract(matching_file, folder_path)
							new_filepath = os.path.join(folder_path, matching_file)

							# Rename the extracted file to remove subfolder structure
							new_filename = os.path.join(folder_path, filename)
							os.rename(new_filepath, new_filename)

							# Remove the now-empty subfolder
							subfolder_path = os.path.dirname(os.path.join(folder_path, matching_file))
							if os.path.exists(subfolder_path) and not os.listdir(subfolder_path):
								os.rmdir(subfolder_path)
							break  # Break the loop if extraction is successful
			except zipfile.BadZipFile:
				print(f"'{zip_file}' is not a valid zip file")
	
def remove_outliers_iqr(data, k=1.5):
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	iqr = q3 - q1
	lower_bound = q1 - k * iqr
	upper_bound = q3 + k * iqr
	filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
	return filtered_data
	
def get_rois(folder_path, rois, sz = False):
	seg_flag, seg = True, []
	for roi in rois:
		if os.path.exists(roi):	seg_name = [roi]
		else:			seg_name = glob(folder_path + '/**/*' + roi + '*.nii*', recursive = True)

		if seg_name: 
			seg.append(nib.load(seg_name[0]).get_fdata())
			if len(seg[-1].shape) < 4: seg[-1] = seg[-1][..., np.newaxis]
		elif sz:
			seg.append(np.zeros(list(sz)+[1]))
		elif len(seg) and seg[-1].size:
			seg.append(np.zeros(list(seg[-1].shape[0:3])+[1]))

	if len(seg): seg = np.concatenate(seg, axis = -1)
	else: seg_flag, seg = False, np.array([])

	return seg_flag, seg

def process_subjects(folder):
	folder_path = os.path.join(directory, folder)
	row = dict()
	hdr = ['B-number', 'Study']
	for key in table.keys():
		row[key] = [folder.split('_')[1],	# B-number
				folder.split('_')[2]]		# study

	for conf in config.keys():
			
		features, labels = config[conf]['data'], config[conf]['data_name'] 
		rois, lesions = config[conf]['rois'], config[conf]['lesions']
		seg_indx, ROI_name = config[conf]['indx'], config[conf]['indx_name']

		seg_flag, seg = get_rois(folder_path, rois)
		
		if seg_flag:	
			les_flag, les = get_rois(folder_path, lesions, seg.shape[0:3])

			if les.size:
				for s_i in range(seg.shape[-1]):
					for l_i in range(les.shape[-1]):
						seg[:,:,:,s_i] = seg[:,:,:,s_i] - binary_dilation(les[:,:,:,l_i], structure = np.ones((3, 3, 3), dtype = np.uint8))
						
				seg = np.concatenate((seg, les), axis = -1)

			for feat in features:
				nii_name = glob(folder_path + '/**/*' + feat + '*.nii*', recursive = True)

				if nii_name:
					nii = nib.load(nii_name[0]).get_fdata()
				else:
					seg_flag = False

				for s_i in range(len(seg_indx)):
					if seg_flag:
						if config[conf]['volumes']:
							row['mean'] = row['mean'] + [np.sum(seg[:,:,:,seg_indx[s_i]] > 0.99)/np.sum(nii > 0.99)]
							row['median'] = row['median'] + row['mean'][-1:]
							row['sd'] = row['sd'] + [0]
						else:
							data = nii[seg[:,:,:,seg_indx[s_i]] > 0.99]
							if len(data): data = remove_outliers_iqr(data)
							row['mean'] = row['mean'] + [np.mean(data)]
							row['median'] = row['median'] + [np.median(data)]
							row['sd'] = row['sd'] + [np.std(data)]
						if (row['mean'][-1] == 0) and (row['median'][-1] == 0) and (row['sd'][-1] == 0):
							for key in table.keys(): row[key][-1] = np.nan
					else:
						row['mean'] = row['mean'] + [np.nan]
						row['median'] = row['median'] + [np.nan]
						row['sd'] = row['sd'] + [np.nan]

		else:
			for key in table.keys():
				row[key] = row[key] + [np.nan]*len(seg_indx)*len(config[conf]['data'])
		
		for lab in labels:
			for s_i in range(len(ROI_name)):
				hdr = hdr + [ROI_name[s_i] + '_' + lab]

	return row, hdr

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Process data files")

	# Optional arguments
	parser.add_argument("-d",	"--directory",	default=os.getcwd(), 	help="Path to subject data folders (default: ./).")
	parser.add_argument("-o",	"--output",	default=None,		help="Prefix for the output file (default: results_YYYYMMDD).")
	parser.add_argument("-t", 	"--table", 	default=None, 		help="Name of the CSV or XLS file to be concatenated to the regional scores (default: None).")
	parser.add_argument("--col",	nargs='+',	default=[], 		help="If table is provided: names of columns to include (default: use all columns).")
	parser.add_argument("--id", 			default=None, 		help="If table is provided: name of the column containing subject IDs (required).")
	parser.add_argument("-u",	"--unzip", 	action="store_true", 	help="Unzip files (default: False).")
	parser.add_argument("-j", 	"--json_config",default=None, 		help="JSON configuration file (required).")
	parser.add_argument("-n", 	"--ncpu", 	default=multiprocessing.cpu_count(),	help="Number of CPUs to use (default: " + str(multiprocessing.cpu_count()) + ")")	

	# Parse the command line arguments
	args = parser.parse_args()

	if args.directory:
		if os.path.exists(args.directory): 
			directory = args.directory 
		else:
			print("Directory", args.directory, "does not exist.")
			exit()
	else:
		directory = os.getcwd()

	# Check if the JSON configuration file is provided
	if args.json_config:
		with open(args.json_config, 'r') as config_file:
			config = json.load(config_file)
		
		if not(config): config = {'IMG_space': {}}
		else: config['IMG_space'] = {}
		
		default_config = {
			'data':		[],
			'rois':		[],
			'lesions': 	[],
			'indx':		[],
			'volumes':	False
		}

		for default_key in default_config.keys():
			if default_key in config.keys():
				config['IMG_space'][default_key] = config[default_key]
				del config[default_key]
	
		if not(config['IMG_space']): del config['IMG_space']
		
		for conf in config.keys():
			config[conf] = set_default_dict(config[conf], default_config)
			config[conf] = check_config(config[conf], conf)
				
	else:
		print("\nERROR: Please provide a JSON configuration file using the -j option.")
		exit()
		
	# Check if the table is provided
	table_xls = False
	if args.table:
		if not os.path.exists(args.table):
			print("\nERROR: The specified file does not exist.")
			exit()

		if args.table.endswith(".csv"):
			# Read CSV file into a DataFrame
			table_xls = pd.read_csv(args.table)
		elif args.table.endswith((".xls", ".xlsx")):
			# Read Excel file into a DataFrame
			table_xls = pd.read_excel(args.table)
		else:
			print("Unsupported file format. Please provide a CSV or Excel file.")
			exit()

		# Filter DataFrame columns based on user-provided column names
		if (args.id) and (args.id in table_xls.columns):
			table_xls = table_xls.loc[table_xls.notna().all(axis=1)]
			table_id = table_xls[args.id]
			table_xls = table_xls.drop(columns = [args.id])
		else:
			print("\nERROR: No subject ID provided or subject ID column name not existing in", args.table)
			exit()			

		# Filter DataFrame columns based on user-provided column names
		if args.col:
			table_xls = table_xls[args.col]

	# Check NCPU parameter
	ncpu = min(multiprocessing.cpu_count(), int(args.ncpu))
	
	# Get the list of folders inside the specified directory
	folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and not name.startswith('.')]

	table = dict()
	table['mean'], table['median'], table['sd'] = [], [], []

	if args.unzip: 
		print('\nUnzipping files...')
		if ncpu > 1: pqdm(folders, unzip_subjects, n_jobs = ncpu)
		else:
			for folder in tqdm(folders):
				unzip_subjects(folder)
		
	print('\nCalculating regional values...')
	if ncpu > 1: results = pqdm(folders, process_subjects, n_jobs = ncpu)
	else:				
		results = []
		for folder in tqdm(folders):
			results.append(process_subjects(folder))

	for key in table.keys(): 
		for result in results:
			row = result[0]
			table[key].append(row[key])

	hdr = results[0][1]
	table_df = dict()
	for key in table.keys():
		table_df[key] = pd.DataFrame(table[key], columns = hdr)

	# Get groups from excel table
	if isinstance(table_xls, pd.DataFrame):
		for key in table.keys():
			for b in table_df[key]['B-number']:
				row = table_xls.loc[table_id.str.contains(b)]
				if len(row):
					for col in table_xls.columns:
							table_df[key].loc[table_df[key]['B-number'] == b, col] = row[col].values[0]
	
	if args.output:
		output = args.output + '_'
	else:
		today = datetime.now().strftime("%Y%m%d")
		output = 'table_' + today + '_'

	for key in table.keys():
		table_df[key].to_csv(output + key + '.csv', index=False)

