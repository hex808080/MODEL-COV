# MODEL-COV
![MODEL_COV_with_legend_](https://github.com/hex808080/MODEL-COV/assets/80628104/7427e997-67f1-486c-8860-4004ce00177f)
## Installation
No installation is required, however it is recommended to use the following conda environment:
- Install conda on your system.
- Run `conda env create -f env.yml` and confirm when prompted.
- Activate `modelcov` enviroment: `conda activate modelcov`

## Make table
- Create your JSON configuration file. Use `make_table_config.json` as reference. The following fields can be listed directly in the JSON file, or can be nested into groups in case of multiple image spaces (as in the example provided):
  - `data`: **required**. Specifies NIFTI files containing maps of interest. Can be a string, list or a dictionary. If dictionary, the keys are used as labels in the final table. File names can be partial.
  - `rois`: **required**. Specifies NIFTI files containing segmentations of ROIs. Can be a string or a list. If list, images after the first must be 3D (NOT 4D). File names can be partial.
  - `lesions`: _optional_. Specifies NIFTI files containing lesions. Dilated lesions (3x3x3 kernel) will be subtracted by segmentation. Lesions will then be added as ROI. If list, images after the first must be 3D (NOT 4D). Can be a string or a list. Lesions will be File names can be partial.
  - `indx`: **required**. Specifies which ROI volumes to use. Can be integer, list or dictionary:
    - _integer_: use all volumes up to the specified value;
    - _list_: use volumes indicated in the list;
    - _dictionary_: use volumes indicated by the dictionary values, and label them as indicated by the dictionary keys.
  -  `volumes`: _optional_. Set as True to calculate ROI volumes instead of mean/median/standard deviation. It assumes a brain mask has been provided as `data`.
- Run `make_table.py`, for example:
  
  ```python -W ignore make_table.py -d ../data -j make_table_config.json -u -t demographics_table.xlsx --id B-number```
  - `-d`: subject folders are in `../data`.
  - `-j`: required files and index details are reported in `make_table_config.json`.
  - `-u`: look for the relevant files inside compressed folders, and unzip them if necessary.
  - `-t`: XLSX or CSV table containing additional information (e.g. demographics) that will be added to the final table.
    - `--id`: **required**. If a table is provided, indicates which column to use to match rows with the corresponding subject folder.
  
  For additional parameters, run `python make_table.py -h`.

## Classify
TBC

## Random Forest
TBC
