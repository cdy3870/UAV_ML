# Predicting UAV Type: An Exploration of Sampling and Data Augmentation for Time Series Classification Code

This repo contains the code for the time series classfication of PX4 flight data as quadrotors, fixed-wings, or hexarotors. 

## Requirements

To reproduce the results from this paper, you first need to set up a virtual environment with the correct dependencies. For simplicity, the conda environment used has been exported and stored as environment.yml. 
Run the following after cloning the repo:

```
conda env create -f environment.yml
conda activate drone_stuff
```

## Source code

1. data_preprocessing.py: contains functions that process the flight review data
2. LSTM_implementation.py: contains the model and accepts arguments to change its configurations and the data used
* This is also where different class imbalance techniques can be applied since the model handles the splitting of data into train and test sets
3. timestamp_tests.py: contains the different timestamp sampling techniques and accepts arguments to change configurations of the sampling

## Reproducing results

### Data

In general, data is obtained and parsed from the raw flight data through the following steps:

1. Run the following to download quad, fixed, and hex data (skip to step 3 if you want to use already parsed data):

```
cd px4-Ulog-Parsers
python data_fetcher.py
```

These commands will download the data into a folder called 'dataDownloaded', containing raw px4 files. You can alter the file to modify the amount of data downloaded and the types of drones.

2. Parsing raw data (approach 1), doing this step allows you to customize features used

```
python data_processing.py -s "parsed_data.txt" -f -1
```

Look at the arguments to see the features that can be used. By default, the features that led to the best performance is used.

This command will produce a mapping of flight ids to dataframes containing the desired features.

3. Using already parsed data (approach 2)

The file 'parsed_data.txt' contains the mapping of flight ids to their dataframes of data using the features that resulted in the best performance (reference paper feats). This data is then binned using the desired sampling method and produces X and y data for the model. 

### Experiments

1. Generating results from parsed data

Parse files are pickled since it takes quite long. Doing so prevents repeating the process many times during the experiments. Once you
have the parsings you can run the bash script provided reproduce the results from the experiments.

```
chmod +x experiments_official_final.sh
source experiments_official_final.sh
```

This script contains all the configurations used in the paper for the different timestamp sampling and class imbalance methods. Create directories called 'official_experiments/sampling' and 'official_experiments//handling_imbalances' to save the full results of the experiments. For the sampling experiment, the parsed file is sampled accordingly and produces an X and y data file, which is fed into the model. Each csv file contains the classification report from each of the 10 folds of the experiment, the standard deviation average, and fold average.
