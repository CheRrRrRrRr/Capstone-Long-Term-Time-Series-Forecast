# Capstone Code: Long term time series forecast on transformer and non-transformer based models
We refer to the structure of the model codes of https://github.com/thuml/Time-Series-Library, and make our own modifications.
## Description of the files

- [x] **data_provider** - Python files of the data factory and data loader that generates batches of data for training, validation, and testing.
- [x] **dataset** - dataset of .csv files. 
- [x] **exp** - Implementation Python file for training, validation and testing.
- [x] **layers** - Layer structures of the models.
- [x] **models** - Main structures of the models.
- [x] **sh_file** - configured .sh files for bashing.
- [x] **utils** - Utility tools for the process of training, validation and testing.
- [x] **create_result_file.py** - Create a "RESULTS" file if it does not exist, and create sub-results files within accordingly.
- [x] **inv_mse.ipynb** - Calculate the MSE metrics for the inverse-standardized result.
- [x] **mse.ipynb** - Calculate the MSE for the CEEMDAN decomposed dataset.
- [x] **run.py** - Configure all the hyper-parameters for model training, validation and testing.

## Run the model

Bash the .sh files in the sh_file directory, and a "RESULTS" and "checkpoints" files will be created. Sub-results with the calculation of MSE for the original data, segments of the result plots and the inver-standardized result will be output to the "RESULTS" file
