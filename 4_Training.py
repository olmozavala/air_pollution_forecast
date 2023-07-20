# %%
import sys
sys.path.append("ai_common_torch/")

from proj_ai.Generators import AirPollutionDataset

# sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home
sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')

import warnings
# Filter the warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

from models.modelselector import 

from viz.figure_generator import plot_input_output_data
from ai_common.constants.AI_params import NormParams, TrainingParams, ModelParams
import ai_common.training.trainingutils as utilsNN
from ai_common.models.modelSelector import select_1d_model
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams

# from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import create_folder
from conf.localConstants import constants
from conf.TrainingUserConfiguration import getTrainingParams
from conf.params import LocalTrainingParams
from proj_io.inout import add_forecasted_hours, add_previous_hours, filter_data, get_column_names, read_merged_files, save_columns
from proj_preproc.preproc import apply_bootstrap, normalizeData

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from os.path import join
import os
 
# %% ========= Set the GPU to use ==================
# In case we want to save the columns to temporal files for debugging purposes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
start_year = 2010
end_year = 2011
hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
forecasted_hours = 24
cur_pollutant = 'otres'
cur_station = 'MER'
val_perc = .1
grid_size = 4
bootstrap = True
boostrap_factor = 10  # Number of times to repeat the bootstrap
boostrap_threshold = 2.9
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


train_dataset = AirPollutionDataset(input_folder, start_year, end_year, model_name, 
                cur_pollutant, output_folder, norm_type,
                forecasted_hours, hours_before, 
                val_perc, 
                bootstrap, boostrap_factor, boostrap_threshold,
                validation_data=False, transform=None)

val_dataset = AirPollutionDataset(input_folder, start_year, end_year, model_name, 
                cur_pollutant, output_folder, norm_type,
                forecasted_hours, hours_before, 
                val_perc, 
                bootstrap, boostrap_factor, boostrap_threshold,
                validation_data=True, transform=None)


print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
workers = 20
train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=1000, num_workers=workers)
print("Done loading data!")

#%% Visualize the data

#%% Initialize the model, loss, and optimizer
model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
                     output_channels=1, start_filters=32, kernel_size=3).to(device)

# TODO how to parallelize the training in multiple GPUs
# n_gpus = torch.cuda.device_count()
# torch.distributed.init_process_group( backend='nccl', world_size=N, init_method='...')
# model = DistributedDataParallel(model, device_ids=[i], output_device=i)

# # criterion = nn.MSELoss()
# loss_func = dice_loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 1000
# model = train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, output_folder)

# #%% Show the results
# # Get a batch of test data
# plot_n = 2
# dataiter = iter(val_loader)
# data, target = next(dataiter)
# data, target = data.to(device), target.to(device)
# output = model(data)
# fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
# for i in range(plot_n):
#     ax[i].imshow(data[i].to('cpu').numpy().squeeze(), cmap='gray')
#     ax[i].set_title(f'True: {target[i]}, Prediction: {output[i].argmax(dim=0)} {[f"{x:0.2f}" for x in output[i]]}', wrap=True)
# plt.show()
# print("Done!")
#
# #%% Showing wrong labels
# ouput_value = output.argmax(dim=1).cpu()
# dif = np.where(ouput_value != target.cpu())[0]
# cur_idx = 0
# #%%
# fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
# for i in range(plot_n):
#     ax[i].imshow(data[dif[cur_idx]].to('cpu').numpy().squeeze(), cmap='gray')
#     ax[i].set_title(f'True: {target[dif[cur_idx]]}, Prediction: {output[dif[cur_idx]].argmax(dim=0)} {[f"{x:0.2f}" for x in output[dif[cur_idx]]]}', wrap=True)
#     cur_idx += 1
# plt.show()
# print("Done!")