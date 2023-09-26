# MODEL-COV
![MODEL_COV_with_legend_](https://github.com/hex808080/MODEL-COV/assets/80628104/7427e997-67f1-486c-8860-4004ce00177f)
## Installation
No installation is required, however it is recommended to use the following conda environment:
- Install conda on your system.
- Run `conda env create -f env.yml` and confirm when prompted.
- Activate `modelcov` enviroment: `conda activate modelcov`

## Make table
### Description
The `make_table.py` function calculates regional scores on an imaging dataset based on the information provided in the JSON configuration file. 

The configuration file allows to easily indicate which maps to calculate the regional score on, which segmentation files to use, which volumes to use as ROI, and more, for one or more image spaces. The function will try to retrieve the data recursively across directories, including zip folders, from full or partial filenames.

Regional scores for each ROI include: **mean**, **median** and **standard** deviation. Regional scores are calculated after removing outliers from the value distribution inside the ROI using the interquartile range method.

Currently, `make_table.py` has been written for data processing at the _Institute of Neurology_ at _University College London_. It presupposes that the dataset is structured at the highest level as a collection of directories, each bearing the name format `XYZ_Bnumber_Study`. In this format, `Bnumber` serves as the unique identifier for each subject, which is used across imaging, clinical and demographic data. The function also assumes that imaging data comprises files with the extensions `.nii` or `.nii.gz`. Nonetheless, it is straightforward to adapt the script to datasets with alternative naming conventions. We are actively developing a more universally applicable version of `make_table.py` to address this variability, and it will be made available in the near future.

### Usage
```python make_table.py [OPTIONS]```

### Options
  - `-j`, `--json_config` (str):  **required**. JSON configuration file containing files and index details (see **Configuration**);
  - `-d`, `--directory` (str):    path where subject folders are located (default: ./);
  - `-o`, `--output` (str):       prefix for the output files (default: "table_YYYYMMDD")
  - `-t`, `--table` (str):        XLS or CSV table containing additional information (e.g. demographics) that will be added to the final table (default: None);
      - `--id` (str):             **required**. If a table is provided, indicates which column to use to match rows with the corresponding subject folder.
      - `--col` (str):            If a table is provided, indicates which column or columns to include (default: all).
  - `-u`, `--unzip`:              look for the relevant files inside compressed folders, and unzip them if found.
  - `-n`, `--ncpu` (int):         number of CPUs to use (default: all available).

### Output
  - `<output>_mean.csv`
  - `<output>_median.csv`
  - `<output>_sd.csv`

### Configuration
Before running `make_table.py`, create your JSON configuration file. Use `make_table_config.json` as reference. The following fields can be listed directly in the JSON file, or can be nested into groups in case of multiple image spaces (as in the example provided):
  - `data`: **required**. Specifies NIFTI files containing maps of interest. Can be a _string_, _list_ or a _dictionary_. If _dictionary_, the keys are used as labels in the final table. File names can be partial.
  - `rois`: **required**. Specifies NIFTI files containing segmentations of ROIs. Can be a _string_ or a _list_. If _list_, images after the first must be 3D (NOT 4D). File names can be partial.
  - `lesions`: _optional_. Specifies NIFTI files containing lesions. Dilated lesions (3x3x3 kernel) will be subtracted from the segmentation. Lesions will then be added as ROI. Can be a _string_ or a _list_. If _list_, images after the first must be 3D (NOT 4D). Lesions will be File names can be partial.
  - `indx`: **required**. Specifies which ROI volumes to use. Can be integer, list or dictionary:
    - _integer_: use all volumes up to the specified value;
    - _list_: use volumes indicated in the list;
    - _dictionary_: use volumes indicated by the dictionary values, and label them as indicated by the dictionary keys.
  -  `volumes`: _optional_. Set as True to calculate ROI volumes instead of mean/median/standard deviation. It assumes a brain mask has been provided as `data`.
  
### Examples
#### Basic usage  
```python -W ignore make_table.py -j make_table_config.json```
#### Advanced usage
```python -W ignore make_table.py -d ../data -j make_table_config.json -u -t demographics_table.xlsx --id B-number```

## Classify
### Description
The `classify.py` script provides a flexible framework for machine learning and statistical data analysis. The function preprocesses the data provided in CSV or XLS format by correcting for covariates (if given), imputing missing values, and standardising the dataset (mean = 0. standard deviation = 1). It then performs any analysis whose implementation has been provided, such as classification or regression. It is highly customizable, with options to specify data columns, groups, covariates, and more.

### Usage
```python classify.py <filename> <labels> [OPTIONS]```

### Input Arguments
  - `<filename>` (str): The input CSV filename containing your dataset.
  - `<labels>` (str): Name of the column containing group labels.

## Random Forest
TBC

## Author
Antonio Ricciardi, PhD. Contact: _antonio.ricciardi@ucl.ac.uk_

## License
TBD
