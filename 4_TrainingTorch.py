# %%
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from proj_ai.TrainingTorch import *
from proj_ai.ModelsTorch import MultilayerPerceptron
from proj_ai.Generators import AirPollutionDataset
from conf.TrainingUserConfigurationTorch import getTrainingParams
from conf.params import LocalTrainingParams 
from ai_common.constants.AI_params import ModelParams, NormParams, TrainingParams
import numpy as np
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize tensorboard
print("Done with imports!")

#%%
# --------------- Dataset parameters ----------------
config = getTrainingParams()
stations = config[LocalTrainingParams.stations]
pollutants = config[LocalTrainingParams.pollutants]
start_year = 2010
end_year = 2015
hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
cur_pollutant = 'otres'
norm_type = NormParams.mean_zero
model_name = config[TrainingParams.config_name]
input_folder = config[TrainingParams.input_folder]
output_folder = config[TrainingParams.output_folder]
val_perc = config[TrainingParams.validation_percentage]
bootstrap = False
bootstrap_factor = 10
bootstrap_threshold = 2.9

date_str = datetime.now().strftime("%Y-%m-%d-%H")
model_name = f"{start_year}_{end_year}_{cur_pollutant}_MP_{date_str}"

# ------------ Model parameters ---------------
model_type = config[ModelParams.MODEL]
batch_normalization = config[ModelParams.BATCH_NORMALIZATION]
dropout = config[ModelParams.DROPOUT]
number_hidden_layers = config[ModelParams.HIDDEN_LAYERS]
cells_per_hidden_layer = config[ModelParams.CELLS_PER_HIDDEN_LAYER]
activation_hidden = config[ModelParams.ACTIVATION_HIDDEN_LAYERS]
activation_output = config[ModelParams.ACTIVATION_OUTPUT_LAYERS]
print("Done with parameters!")

# %% ------ Initialize the dataset and dataloaders ------
dataset = AirPollutionDataset(input_folder=input_folder,start_year=start_year, 
                              end_year=end_year, model_name='', 
                              cur_pollutant=cur_pollutant, output_folder=output_folder, 
                              norm_type=norm_type, forecasted_hours=24, hours_before=hours_before, 
                              val_perc=val_perc, bootstrap=bootstrap, boostrap_factor=bootstrap_factor, 
                              boostrap_threshold=bootstrap_threshold) 

# Get the training and validation indexes
train_idxs = dataset.get_train_idxs()
val_idxs = dataset.get_val_idxs()

train_dataset = torch.utils.data.Subset(dataset, train_idxs)
val_dataset = torch.utils.data.Subset(dataset, val_idxs)

# Create DataLoaders for training and validation
workers = 10
shuffle_data = False
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=shuffle_data, num_workers=workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=shuffle_data, num_workers=workers)
print("Done loading data!")

# %% ------------------Plot some data ------------------
# # Get a batch of training data
dataiter = iter(train_loader)
data, target = next(dataiter)
x_names = dataset.get_x_names()
y_names = dataset.get_y_names()
# %%
x = [x[0] for x in data[0:72]]
y = [x[0] for x in target[0:72]]
plt.plot(x, label='Input')
plt.plot(y, 'r--', label='Output') 
plt.legend()
plt.show()

# %% Initialize the model, loss, and optimizer
input_size = dataset.get_input_size()
output_layer_size= dataset.get_output_size()

model = MultilayerPerceptron(input_size, number_hidden_layers, cells_per_hidden_layer, 
                                 output_layer_size, batch_norm=batch_normalization, dropout=dropout,
                                activation_hidden=activation_hidden,
                                activation_output=activation_output).to(device)

# RMS criterion
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Done initializing model!")


# %%
num_epochs = 50
model = train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, model_name, output_folder)

#%% Show the results
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
# # %%
