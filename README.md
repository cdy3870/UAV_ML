# Predicting UAV Type: An Exploration of Sampling and Data Augmentation for Time Series Classification Code

This repo contains the code for the time series classfication of PX4 flight data as quadrotors, fixed-wings, or hexarotors. 

## Requirements

To reproduce the results from this paper, you first need to set up a virtual environment with the correct dependencies. For simplicity, the conda environment used has been exported and stored as environment.yml. 
Run the following after cloning the repo:

```
conda env create -f environment.yml
```

Then activate the environment using:

```
conda activate drone_stuff
```

## Data


## Source code

1. data_preprocessing.py: contains functions that process the flight review data
2. LSTM_implementation.py: contains the model and accepts arguments to change its configurations and the data used
* This is also where different class imbalance techniques can be applied since the model handles the splitting of data into train and test sets
3. timestamp_tests.py: contains the different timestamp sampling techniques and accepts arguments to change configurations of the sampling

## Reproducing results

### Data

In general, data is obtained and parsed from the raw flight data through the following steps:

1. Run the following to download quad, fixed, and hex data:

```
cd px4-Ulog-Parsers
python TarikDataFetcher.py
```

These commands will download the data into a folder called 'dataDownloaded', containing raw px4 files. You can alter the file to modify the amount of data downloaded and the types of drones.

2. Parsing raw data (approach 1), doing this step allows you to customize features used

```
python data_processing.py
```

This command will produce a mapping of flight ids to dataframes containing the desired features.

3. Using already parsed data (approach 2), doing this step instead of 2 will lead to the results from the experiments



4. Generating results from parsed data

Parse files are pickled since it takes quite long. Doing so prevents repeating the process many times during the experiments. Once you
have the parsings you can run the bash script provided reproduce the results from the experiments.

```
chmod +x experiments_official_final.sh
source experiments_official_final.sh
```

This script contains all the configurations used in the paper for the different timestamp sampling and class imbalance methods.
