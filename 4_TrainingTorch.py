
# %%
import sys
sys.path.append('./eoas_pyutils')

from ai_common.constants.AI_params import NormParams, TrainingParams, ModelParams
import ai_common.training.trainingutils as utilsNN
from ai_common.models.modelSelector import select_1d_model
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams

from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import create_folder
from conf.localConstants import constants
from conf.TrainingUserConfiguration import getTrainingParams
from conf.params import LocalTrainingParams, PreprocParams

from datetime import date, datetime, timedelta
# from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from os.path import join
import matplotlib.pyplot as plt
import os
import time

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel


# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = getTrainingParams()
stations = config[LocalTrainingParams.stations]
pollutants = config[LocalTrainingParams.pollutants]
start_year = 2010
end_year = 2013
validation_year = 2013
hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
cur_pollutant = 'otres'

input_folder = config[TrainingParams.input_folder]
output_folder = config[TrainingParams.output_folder]

output_folder = join(output_folder,F"MultipleStations_MultiplePollutants_{start_year}_{end_year}")

val_perc = config[TrainingParams.validation_percentage]
test_perc = config[TrainingParams.test_percentage]
eval_metrics = config[TrainingParams.evaluation_metrics]
loss_func = config[TrainingParams.loss_function]
batch_size = config[TrainingParams.batch_size]
epochs = config[TrainingParams.epochs]
model_name_user = config[TrainingParams.config_name]
optimizer = config[TrainingParams.optimizer]
forecasted_hours = config[LocalTrainingParams.forecasted_hours]
norm_type = config[TrainingParams.normalization_type]

split_info_folder = join(output_folder, 'Splits')
parameters_folder = join(output_folder, 'Parameters')
weights_folder = join(output_folder, 'models')
logs_folder = join(output_folder, 'logs')
imgs_folder= join(output_folder, 'imgs')
norm_folder = join(output_folder, 'norm')
create_folder(split_info_folder)
create_folder(parameters_folder)
create_folder(weights_folder)
create_folder(logs_folder)
create_folder(norm_folder)

# %% Reading the data
input_folder = config[TrainingParams.input_folder]
# -------- Reading all the years in a single data frame (all stations)
for c_year in range(start_year, end_year+1):
    db_file_name = join(input_folder, F"{c_year}_AllStations.csv") # Just for testing
    print(F"============ Reading data for: {c_year}: {db_file_name}")
    if c_year == start_year:
        data = pd.read_csv(db_file_name, index_col=0)
    else:
        data = pd.concat([data, pd.read_csv(db_file_name, index_col=0)])
print("Done!")

# %% -------- 
config[ModelParams.INPUT_SIZE] = len(data.columns)
print(F'Data shape: {data.shape} Data axes {data.axes}')
print("Done!")

datetimes_str = data.index.values
datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

# %% -------- Normalizing data
print("Normalizing data....")
if norm_type == NormParams.min_max:
    scaler = preprocessing.MinMaxScaler()
if norm_type == NormParams.mean_zero:
    scaler = preprocessing.StandardScaler()

scaler = scaler.fit(data)
data_norm_np = scaler.transform(data)
data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)
print(F'Done!')

# %% Filtering only dates where there is data "forecasted hours after" (24 hrs after)
print(F"Building X and Y ....")
hours_before = 3 # How many hours of pollution data are we adding as input to the model (current - hours_before)

# For X we need to remove all the columns of the stations and the last hours
myregex = f"cont_.*"
contaminant_columns = data_norm_df.filter(regex=myregex).columns
# This dataframe contains all the columns of the contaminants for all stations 

# Adding the previous hours of the pollutants as extra columns
print(F'{data_norm_df.shape}')
X_df = data_norm_df.copy()
for c_hour in range(1, hours_before+1):
    for c_column in contaminant_columns:
        X_df[f'minus_{c_hour:02d}_{c_column}'] = data_norm_df[c_column].shift(-c_hour)

# Adding the forecasted hours of the pollutants as extra columns (specific contaminant)
myregex = f"cont_{cur_pollutant}.*"
single_cont_columns = data_norm_df.filter(regex=myregex).columns
Y_df = data_norm_df.loc[:, data_norm_df.columns.isin(single_cont_columns)].copy()
for c_hour in range(1, forecasted_hours+1):
    for c_column in single_cont_columns:
        Y_df[f'plus_{c_hour:02d}_{c_column}'] = Y_df[c_column].shift(c_hour)

X_df = data_norm_df.iloc[hours_before:,:]
Y_df = Y_df.iloc[hours_before:,:]
print("Done!")

# %%
print(F'Original {data_norm_df.shape}')
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
print(F'Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')

#%% Split the training data by year
print("Splitting training and validation data by year....")
train_idxs = X_df.index <= F"{validation_year}-01-01 00:00:00"
val_idxs = X_df.index > F"{validation_year}-01-01 00:00:00"

X_df_train = X_df[train_idxs]
Y_df_train = Y_df[train_idxs]
X_df_val = X_df[val_idxs]
Y_df_val = Y_df[val_idxs]

print(F'X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
print(F'Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
print(F'X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum()/1024**2:02f} MB')
print(F'Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum()/1024**2:02f} MB')
# %%

# Here we remove the datetime indexes so we need to consider that 
X_df_train.reset_index(drop=True, inplace=True)
Y_df_train.reset_index(drop=True, inplace=True)
X_df_val.reset_index(drop=True, inplace=True)
Y_df_val.reset_index(drop=True, inplace=True)

print("Done!")

# %% -------- Bootstrapping the data
def apply_bootstrap(X_df, Y_df, contaminant, station, boostrap_threshold, forecasted_hours, boostrap_factor=1):
    '''
    This function will boostrap the data based on the threshold and the forecasted hours
    '''
    bootstrap_column = f"cont_{contaminant}_{station}"
    print("Bootstrapping the data...")
    bootstrap_idx = X_df.loc[:,bootstrap_column] > boostrap_threshold
    # Searching all the index where X or Y is above the threshold
    for i in range(1, forecasted_hours+1):
        # print(bootstrap_idx.sum())
        bootstrap_idx = bootstrap_idx | (Y_df.loc[:,f"plus_{i:02d}_{bootstrap_column}"] > boostrap_threshold)

    X_df = pd.concat([X_df, *[X_df[bootstrap_idx] for i in range(boostrap_factor)]])
    Y_df = pd.concat([Y_df, *[Y_df[bootstrap_idx] for i in range(boostrap_factor)]])

    return X_df, Y_df

bootstrap = True
boostrap_factor = 3  # Number of times to repeat the bootstrap
boostrap_threshold = 2.9
if bootstrap:
    # -------- Bootstrapping the data
    station = "MER"
    print(F'X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
    X_df_train, Y_df_train = apply_bootstrap(X_df_train, Y_df_train, cur_pollutant, station, boostrap_threshold, forecasted_hours, boostrap_factor)
    print(F'X train bootstrapped {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y train bootstrapped {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum()/1024**2:02f} MB')

# %% 
print(f"Train examples: {X_df_train.shape[0]}")
print(f"Validation examples {X_df_val.shape[0]}")

print("Selecting and generating the model....")
now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = F'{model_name_user}_{cur_pollutant}_{now}'

# print(F"Norm params: {scaler.get_params()}")
# file_name_normparams = join(parameters_folder, F'{model_name}.csv')
# utilsNN.save_norm_params(file_name_normparams, norm_type, scaler)

# ******************* Selecting the model **********************
config[ModelParams.INPUT_SIZE] = X_df_train.shape[1]
config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = Y_df_train.shape[1]


# file_name_splits = join(split_info_folder, F'{model_name}.csv')
# info_splits = DataFrame({F'Train({len(train_ids)})': train_ids})
# info_splits[F'Validation({len(val_ids)})'] = 0
# info_splits[F'Validation({len(val_ids)})'][0:len(val_ids)] = val_ids
# info_splits[F'Test({len(test_ids)})'] = 0
# info_splits[F'Test({len(test_ids)})'][0:len(test_ids)] = test_ids
# info_splits.to_csv(file_name_splits, index=None)

# print(F"Norm params: {scaler.get_params()}")
# file_name_normparams = join(parameters_folder, F'{model_name}.csv')
# scaler.path_file = join(norm_folder,F"{model_name}_scaler.pkl")  #path_file_name to save the pickled scaler
# utilsNN.save_norm_params(file_name_normparams, norm_type, scaler)
# info_splits.to_csv(file_name_splits, index=None)

#%% Initialize the model, loss, and optimizer
model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
                     output_channels=1, start_filters=32, kernel_size=3).to(device)

# TODO how to parallelize the training in multiple GPUs
# n_gpus = torch.cuda.device_count()
# torch.distributed.init_process_group( backend='nccl', world_size=N, init_method='...')
# model = DistributedDataParallel(model, device_ids=[i], output_device=i)

# criterion = nn.MSELoss()
loss_func = dice_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
model = train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, output_folder)