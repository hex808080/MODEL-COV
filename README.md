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
- `-j`, `--json_config` (str):  **required**. JSON configuration file containing files and index details (see **Configuration**).
- `-d`, `--directory` (str):    Path where subject folders are located (default: ./).
- `-o`, `--output` (str):       Prefix for the output files (default: "table_YYYYMMDD").
- `-t`, `--table` (str):        XLS or CSV table containing additional information (e.g. demographics) that will be added to the final table (default: None).
    - `--id` (str):             **required**. If a table is provided, indicates which column to use to match rows with the corresponding subject folder.
    - `--col` (str):            If a table is provided, indicates which column or columns to include (default: all columns).
- `-u`, `--unzip`:              Look for the relevant files inside compressed folders, and unzip them if found.
- `-n`, `--ncpu` (int):         Number of CPUs to use (default: all available CPUs).

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

### Options
- `-g`, `--groups` (str): Names of the groups to classify (default: use all groups).
- `-c`, `--covariates` (str): Name of the columns containing covariates to be regressed out from the dataset (default: None).
- `-r`, `--reference` (str): Reference group(s) on which to calculate regression coefficients. Only used if covariates are provided (default: regress over the whole dataset).
- `-e`, `--exclude` (str): Name of the columns to exclude from the dataset (default: None).
- `-k`, `--keep` (str): Name of the columns to keep from the dataset (default: None).
- `-a`, `--algorithm` (str): Name of the algorithm to use. A Python function with the same name must be available for import and contain a `run(y, X, config)` method (see **Notes**) (default: random_forest).
- `-j`, `--json_config` (str): Name of a JSON file containing additional parameters for classification and formatting (default: None).
- `-i`, `--intermediates` (str): Name of the folder storing intermediate files (default: None).
- `-o`, `--output` (str): Name prefix of output files (default: results_YYYYMMDD).
- `-n`, `--ncpu` (int): Number of CPUs to use (default: all available CPUs).

### Outputs
By default, the script will produce output files with filenames starting with `results_YYYYMMDD`, where YYYYMMDD represents the current date, and the specifics of the outputs will depend on the algorithm chosen via the `-a` option (see **Random Forest**). If the `-i` option is used, the following intermediate files will also be saved in the specified directory:
  - `table_corrected.csv`: table after correcting for covariates, if specified;
  - `table_corrected_beta.csv`: log of the regression reporting significant regression coefficients and associated p-values.
  - `table_impstd.csv`: table after imputing for missing values and value standardisation.

### Examples
#### Basic Usage
```python -W ignore classify.py table_20230924_mean_fix.csv Pt_group_label```
#### Advanced Usage
```python -W ignore classify.py table_20230924_mean_fix.csv Pt_group_label --json_config classify_config.json --exclude B-number Study Project Project_label Gender Pt_group cGM_FA dGM_FA WM_QSM cGM_QSM BS_QSM --covariates Age Gender_label --reference HC -i intermediates --groups HC Long\ COVID```

### Notes
- Correction for covariates is performed using 2nd order least square. If the `-r` option is used, the regression coefficients will be calculated only on the spedified group, and then applied to the entire dataset. This is particularly useful when grouops and covariates are coupled, e.g. patients are also older than healthy controls, and regressing for age would also regress out information about pathology. By specifying the healthy controls group via `-r`, only healthy controls' data is used to infer the effect of age on the data, and the resulting regression coefficients are then used to correct for age across the whole dataset.
- Make sure that any algorithm specified with `-a` can be imported from the current working directory. The algorithm implementation **MUST** also contain the method `run(y, X, config)`. The datatype of the input variables (e.g. _list_, _numpy array_, _dictionary_, _pandas dataframe_, etc.) and how the variables are used, depend entirely on the algorithm:
  - `y`:  variable containing group labels (for classification) or independent variable (for regression);
  - `X`:  variable containing features (for classification) or dependent variables (for regression);
  - `config`: variable containing any additional parameter.
- Use case: wrap any algorithm you may have access to into a `run(y, X, config)` function, defining the input variables accordingly. Save the script in the current working directory as `my_algorithm.py`, and then launch `classify.py` with the option `-a my_algorithm`. A JSON configuration file for the specific algorithm can be provided via the `-j` option. This allows the user to expand the functionality of `classify.py` without directly affecting the code. Additional algorithms for machine learning and/or regression may be added to this repository in the future.

## Random Forest
### Description
The `random_forest.py` script is the default algorithm for the `classify.py` function. It uses random forest to perform binary or multi-label classification, returning multiple classification performance scores, and reports the results into a combined bar- and box-plot that clearly shows feature importance and value distributions across groups. A JSON configuration file can be provided to highly customise the classification and plotting, with options to add a randomisation step to calculate the significance of the classification performance scores, custom palettes for feature groupings, and more (see **Configuration**).

### Usage
```
from random_forest import *
results = run(y, X, config)
```

### Input Arguments
- `y`: _Pandas dataframe_ containing group labels.
- `X`: _Pandas dataframe_ containing features.
- `config`: (_optional_) _dictionary_ containing configuration parameters.

### Outputs
The script will produce output files with filenames prefix determied by the parent function:
- `*.json`: JSON file containing all radom forest results, e.g. classification performance per iteration, feature importance, etc. 
- `*.png`: Combined bar- and box-plots reporting ranked feature importance and group values distrinbutions.
- `*.mpl`: If the `mpl_save` option is used in the configuration file (see **Configuration**). _Matplotlib_ object containing the combined bar- and box-plots.

### Configuration
The random_forest algorithm configuration file is divided into two main sections: `random_forest_config` and `random_forest_plot_config`. These sections allow you to fine-tune the behavior of the random forest classifier and customize the appearance of the associated bar- and box-plots. Below are the available configuration options (see `classify_config.json` file):

#### random_forest_config
- `N_trees`: Number of trees in the random forest (default: 1000).
- `N_split`: Number of K-fold splits (default: 10).
- `N_iter`: Number of stratified iterations (default: 100).
- `N_shuffle`: Number of label-shuffling iterations to determine the significance of classification performance scores (default: 0; recommended: 100).
- `resampling`: Type of resampling method to use in case of unbalanced data (default: under):
    - `nores`: No resampling.
    - `under`: Randomly undersample the majority class to match the minority class.
    - `SMOTE`: Oversample the minority class to match the majority class via _Synthetic Minority Oversampling Technique_.
- `random_state`: Random seed for reproducibility (default: 42).

#### random_forest_plot_config
- `groups`: Color-group features in the bar plot, allowing you to color bars associated with features within the same group with the same color. The legend will display the corresponding group labels (default: no grouping). You can specify partial feature names, and in case of ambiguity, use the # special character for precise matching.
- `fontsize`: Font size in points to use in the plot (default: 12).
- `figsize`: Size of the figure in inches to use in the plot (default: [10, 6]).
- `palette_bar`: Palette to use for the bar plot. It accepts either a string (name of an existing Matplotlib palette) or a custom list of RGB values (default: tab10). For custom lists, you can define your own color scheme.
- `palette_box`: Palette to use for the box plot. Similar to palette_bar, it accepts either a string or a custom list (default: Pastel2).
- `N_bars`: Number of features to include in the plot (default: -1 for all features).
- `mpl_save`: Save the image as an object for further customization (default: false).

The parent function (`classify.py`) will add additional fields to the configuration file. **DO NOT** use these keywords for new algorithm scripts:
- `algorithm`: name of the algorithm to use;
- `output`: output filename prefix;
- `intermediates`: name of the folder containing intermediate files produced by the parent function;
- `ncpu`: number of CPUs to use.
  
## Author
Antonio Ricciardi, PhD. Contact: _antonio.ricciardi@ucl.ac.uk_

## License
TBD
