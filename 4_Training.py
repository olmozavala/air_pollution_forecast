# %%
import sys

from viz.figure_generator import plot_input_output_data
# sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home
sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')

import warnings
# Filter the warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

from ai_common.constants.AI_params import NormParams, TrainingParams, ModelParams
import ai_common.training.trainingutils as utilsNN
from ai_common.models.modelSelector import select_1d_model
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams

# from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import create_folder
from conf.localConstants import constants
from conf.TrainingUserConfiguration import getTrainingParams
from conf.params import LocalTrainingParams, PreprocParams
from proj_io.inout import add_forecasted_hours, add_previous_hours, filter_data, get_column_names, read_merged_files, save_columns
from proj_preproc.preproc import apply_bootstrap, normalizeData

from datetime import date, datetime, timedelta
import tensorflow as tf
# from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from os.path import join
import matplotlib.pyplot as plt
import os
import time
import pickle
 

# %% ========= Set the GPU to use ==================
# In case we want to save the columns to temporal files for debugging purposes
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
config = getTrainingParams()
stations = config[LocalTrainingParams.stations]
pollutants = config[LocalTrainingParams.pollutants]
start_year = 2010
end_year = 2013
hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
cur_pollutant = 'otres'
cur_station = 'MER'

input_folder = config[TrainingParams.input_folder]
output_folder = config[TrainingParams.output_folder]

output_folder = join(output_folder,F"MultipleStations_MultiplePollutants_{start_year}_{end_year}")

bootstrap = True
boostrap_factor = 10  # Number of times to repeat the bootstrap
boostrap_threshold = 2.9

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
create_folder(imgs_folder)

# %% Reading the data
input_folder = config[TrainingParams.input_folder]
data = read_merged_files(input_folder, start_year, end_year)
config[ModelParams.INPUT_SIZE] = len(data.columns)

datetimes_str = data.index.values
datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

# %% -------- Normalizing data
now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = F'{model_name_user}_{cur_pollutant}_{now}'
file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  
print("Normalizing data....")
data_norm_df = normalizeData(data, norm_type, file_name_norm)

# %% ====== Getting all the orignal columns by type
all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(data_norm_df)

# %% ====== Remove columns for other pollutants
# Here we remove all the data of other pollutants
X_df = filter_data(data_norm_df, filter_type='single_pollutant',
                   filtered_pollutant=cur_pollutant) 

print(X_df.columns.values)
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

# %% ====== Adding the previous hours of the pollutants as extra columns (all contaminants)
print(F"Building X and Y ....")
X_df = add_previous_hours(X_df, hours_before=24)

# %% ====== Adding the forecasted hours of the pollutants as the predicted column Y (specific contaminant)
print("\tAdding the forecasted hours of the pollutant as the predicted column Y...")
Y_df = add_forecasted_hours(X_df, cur_pollutant, range(1,forecasted_hours+1))

# %% Remove the first hours because Print the final shape of X and Y
X_df = X_df.iloc[hours_before:,:]
Y_df = Y_df.iloc[hours_before:,:]
save_columns(Y_df, join(output_folder, 'Y_columns.csv'))
save_columns(X_df, join(output_folder, 'X_columns.csv'))
print("Done!")

print(F'Original {data_norm_df.shape}')
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
print(F'Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')

#%% Split the training data by year
print("Splitting training and validation data by year....")

splits_file = join(split_info_folder, F'splits_{model_name}.csv')
train_idxs, val_idxs, test_idxs = utilsNN.split_train_validation_and_test(
    len(X_df), val_perc, test_perc, shuffle_ids=False, file_name=splits_file)


# %% Remove time as index 
# Here we remove the datetime indexes so we need to consider that 
print("Removing time index...")
X_df.reset_index(drop=True, inplace=True)
Y_df.reset_index(drop=True, inplace=True)

X_df_train = X_df.iloc[train_idxs]
Y_df_train = Y_df.iloc[train_idxs]

X_df_val = X_df.iloc[val_idxs]
Y_df_val = Y_df.iloc[val_idxs]

print(F'X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
print(F'Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
print(F'X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum()/1024**2:02f} MB')
print(F'Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum()/1024**2:02f} MB')

print("Done!")

# %% ======= Bootstrapping the data
if bootstrap:
    # -------- Bootstrapping the data
    # Se utiliza esta estacion para decidir que indices son los que se van a usar para el bootstrapping.
    # Only the indexes for this station that are above the threshold will be used for bootstrapping
    station = "MER" 
    print("Bootstrapping the data...")
    print(F'X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
    X_df_train, Y_df_train = apply_bootstrap(X_df_train, Y_df_train, cur_pollutant, station, boostrap_threshold, forecasted_hours, boostrap_factor)
    print(F'X train bootstrapped {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y train bootstrapped {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum()/1024**2:02f} MB')
    print(F'X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum()/1024**2:02f} MB')
    print(F'Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum()/1024**2:02f} MB')


#%% Replace all the nan values with another value
replace_value = 0
print(f"Replacing nan values with {replace_value}...")
X_df_train.fillna(replace_value, inplace=True)
X_df_val.fillna(replace_value, inplace=True)
Y_df_train.fillna(replace_value, inplace=True)
Y_df_val.fillna(replace_value, inplace=True)

# %% Visualize input and outputs
print("Visualizing input and outputs...")
plot_input_output_data(X_df_train, Y_df_train, cur_station, cur_pollutant, 
                       imgs_folder, model_name)

# %% 
print(f"Train examples: {X_df_train.shape[0]}")
print(f"Validation examples {X_df_val.shape[0]}")

# ******************* Selecting the model **********************
config[ModelParams.INPUT_SIZE] = X_df_train.shape[1]
config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = Y_df_train.shape[1]

model = select_1d_model(config)
print("Done!")


#print("Getting callbacks ...")
all_callbacks = utilsNN.get_all_callbacks(model_name=model_name,
                                           early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                             weights_folder=weights_folder,
                                             patience=50,
                                             logs_folder=logs_folder)

print("Compiling model ...")
model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)
model.summary()

#%% 

print("Training ...")
model.fit(X_df_train.values, Y_df_train.values,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_df_val.values, Y_df_val.values),
                    shuffle=True,
                    callbacks=all_callbacks)
