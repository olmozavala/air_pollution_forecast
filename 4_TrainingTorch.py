# %%
import sys
sys.path.append('./eoas_pyutils')

from ai_common.constants.AI_params import NormParams, TrainingParams, ModelParams
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams

from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import create_folder
from proj_io.inout import read_merged_files
from conf.localConstants import constants

from datetime import date, datetime, timedelta
# from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from pandas import DataFrame
from os.path import join
import matplotlib.pyplot as plt
import os
import time
# from proj_ai.Generators.AirPollutionDataset
from proj_preproc.preproc import normalizeData

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel

# %% ========= Set the GPU to use ==================
# In case we want to save the columns to temporal files for debugging purposes
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# %%
start_year = 2014
end_year = 2017
validation_year = 2017
hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
forecasted_hours = 24
cur_pollutant = 'otres'
cur_station = 'MER'
grid_size = 4
merged_specific_folder = f'{grid_size*grid_size}' # We may have multiple folders inside merge depending on the cuadrants

now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name_user = "Torch_Test1"
model_name = F'{model_name_user}_{cur_pollutant}_{now}'

data_folder = '/ZION/AirPollutionData/Data/'

input_folder = join(data_folder, "MergedDataCSV", merged_specific_folder)
output_folder = join(data_folder, 'TrainingTestsOZ', 
                     F"MultipleStations_MultiplePollutants_{start_year}_{end_year}")

batch_size = 5000
norm_type = NormParams.mean_zero

# %%
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
data = read_merged_files(input_folder, start_year, end_year)

# %% -------- Normalizing data
print("Normalizing data....")
file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  
data_norm_df = normalizeData(data, norm_type, file_name_norm)

# %% ====== Getting all the orignal columns by type
myregex = f"cont_.*"
all_contaminant_columns = data_norm_df.filter(regex=myregex).columns
# print(all_contaminant_columns.values)
all_time_colums = data_norm_df.filter(regex="day|year|week").columns
# print(all_time_colums.values)
all_meteo_columns = [x for x in data_norm_df.columns if x not in all_contaminant_columns and x not in all_time_colums]
# print(all_meteo_columns)

# %% ====== Remove columns for other pollutants
if True: # In case we want to use a single pollutant and station
    # ------------- Here we only keep the columns for the current station and pollutant
    # keep_cols = [f'cont_{cur_pollutant}_{cur_station}'] + all_time_colums.tolist() + all_meteo_columns
    # print(F"Keeping columns: {len(keep_cols)} original columns: {len(data_norm_df.columns)}")
    # X_df = data_norm_df[keep_cols].copy()

    # ---------- Here we only keep the columns for the current pollutant all stations
    keep_cols = [x for x in data_norm_df.columns if x.startswith(f'cont_{cur_pollutant}')] + all_time_colums.tolist() + all_meteo_columns
    print(F"Keeping columns: {len(keep_cols)} original columns: {len(data_norm_df.columns)}")
    X_df = data_norm_df[keep_cols].copy()
else:
    X_df = data_norm_df.copy()
print(X_df.columns.values)

print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

# %% ====== Adding the previous hours of the pollutants as extra columns (all contaminants)
print(F"Building X and Y ....")
print("\tAdding the previous hours of the pollutants as additional columns...")
myregex = f"cont_.*"
contaminant_columns = X_df.filter(regex=myregex).columns
print(F"\t\tContaminant columns: {contaminant_columns.values}")
for c_hour in range(1, hours_before+1):
    for c_column in contaminant_columns:
        X_df[f'minus_{c_hour:02d}_{c_column}'] = X_df[c_column].shift(c_hour)
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
print("Done!")


# %% ====== Adding the forecasted hours of the pollutants as the predicted column Y (specific contaminant)
print("\tAdding the forecasted hours of the pollutant as the predicted column Y...")
myregex = f"^cont_{cur_pollutant}.*"
single_cont_columns = X_df.filter(regex=myregex).columns
print(single_cont_columns)

# Adds the next 24 (forecasted_hours) hours to the prediction
Y_df =  pd.DataFrame(index=data_norm_df.index)
# for c_hour in range(forecasted_hours, forecasted_hours+1):  # In case you want to add only one hour
for c_hour in range(1, forecasted_hours+1):
    for c_column in single_cont_columns:
        Y_df[f'plus_{c_hour:02d}_{c_column}'] = X_df[c_column].shift(-c_hour)

print(f"Shape of Y: {Y_df.shape}")
save_columns(Y_df, 'Y')
# %% Print the final shape of X and Y
X_df = X_df.iloc[hours_before:,:]
Y_df = Y_df.iloc[hours_before:,:]
save_columns(X_df, 'X')
save_columns(Y_df, 'Y')
print("Done!")

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

# %% Remove time as index 
# Here we remove the datetime indexes so we need to consider that 
print("Removing time index...")
X_df_train.reset_index(drop=True, inplace=True)
Y_df_train.reset_index(drop=True, inplace=True)
X_df_val.reset_index(drop=True, inplace=True)
Y_df_val.reset_index(drop=True, inplace=True)

print("Done!")

# %% ======= Bootstrapping the data
def apply_bootstrap(X_df, Y_df, contaminant, station, boostrap_threshold, forecasted_hours, boostrap_factor=1):
    '''
    This function will boostrap the data based on the threshold and the forecasted hours
    '''

    bootstrap_column = f"cont_{contaminant}_{station}"
    print("Bootstrapping the data...")
    # Searching all the index where X or Y is above the threshold

    # Adding index when the current time is above the threshold
    bootstrap_idx = X_df.loc[:,bootstrap_column] > boostrap_threshold

    # Searching index when any of the forecasted hours is above the threshold
    y_cols = Y_df.columns.values
    for i in range(1, forecasted_hours+1):
        # print(bootstrap_idx.sum())  
        c_column = f"plus_{i:02d}_{bootstrap_column}"
        if c_column in y_cols:
            bootstrap_idx = bootstrap_idx | (Y_df.loc[:, c_column] > boostrap_threshold)

    X_df = pd.concat([X_df, *[X_df[bootstrap_idx] for i in range(boostrap_factor)]])
    Y_df = pd.concat([Y_df, *[Y_df[bootstrap_idx] for i in range(boostrap_factor)]])

    return X_df, Y_df

bootstrap = True
boostrap_factor = 10  # Number of times to repeat the bootstrap
boostrap_threshold = 2.9
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
# -------- Visualizing the input and output
def addColumn(col_name, start_idx, end_idx, df, ax, size=20, scatter=False):
    if scatter:
        ax.scatter(df.index[start_idx:end_idx], 
               df[col_name][start_idx:end_idx], 
               label=col_name, s=size)
    else:
        ax.plot(df.index[start_idx:end_idx], 
               df[col_name][start_idx:end_idx], 
               label=col_name)

fig, ax = plt.subplots(1,3, figsize=(30,10))
station = cur_station
times_to_plot = 48
start_idx = 104
end_idx = start_idx + times_to_plot
addColumn(f"cont_{cur_pollutant}_{station}", start_idx, end_idx, X_df_train, ax[0], scatter=False)
# Add the predicted values Y (next 24 hours)
# for c_hour in range(forecasted_hours, forecasted_hours+1):
# for c_hour in range(forecasted_hours-5, forecasted_hours+1):
for c_hour in range(1, 5):
    addColumn(f"plus_{c_hour:02d}_cont_{cur_pollutant}_{station}", 
              start_idx, end_idx, Y_df_train, ax[0], size=10)

# for c_hour in range(1, 5):
    # addColumn(f"minus_{c_hour:02d}_cont_{cur_pollutant}_{station}", 
            #   start_idx, end_idx, X_df_train, ax[0], size=10)

# Plot some meteo columns
meteo_col = "T2"
plot_hr = 0
cuadrants = 4
tot_cuadrants = int(cuadrants**2 )
cols = [f"{meteo_col}_{i}_h{plot_hr}" for i in range(tot_cuadrants)]
meteo_data = X_df_train.loc[start_idx, cols]
meteo_img = np.zeros((cuadrants, cuadrants))
# Fill the meteo image with the data from the dataframe
for i in range(cuadrants):
    for j in range(cuadrants):
        meteo_img[i,j] = meteo_data[f"{meteo_col}_{i*cuadrants+j}_h{plot_hr}"]

ax[2].imshow(meteo_img, cmap='hot', interpolation='nearest')

ax[0].legend()
# Add some of the time columns
addColumn(f"sin_day", start_idx, end_idx, X_df, ax[1], size=10)
addColumn(f"cos_day", start_idx, end_idx, X_df, ax[1], size=10)
addColumn(f"half_sin_day", start_idx, end_idx, X_df, ax[1], size=10)

plt.show()
plt.savefig('input.png')
plt.close()
# %%
scaler.data_min_()
# %% 
print(f"Train examples: {X_df_train.shape[0]}")
print(f"Validation examples {X_df_val.shape[0]}")

# ******************* Selecting the model **********************
config[ModelParams.INPUT_SIZE] = X_df_train.shape[1]
config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = Y_df_train.shape[1]

model = select_1d_model(config)
print("Done!")


#%% 


#%% Initialize the model, loss, and optimizer
# model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
#                      output_channels=1, start_filters=32, kernel_size=3).to(device)

# # TODO how to parallelize the training in multiple GPUs
# # n_gpus = torch.cuda.device_count()
# # torch.distributed.init_process_group( backend='nccl', world_size=N, init_method='...')
# # model = DistributedDataParallel(model, device_ids=[i], output_device=i)

# # criterion = nn.MSELoss()
# loss_func = dice_loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 1000
# model = train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, output_folder)