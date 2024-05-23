#!/usr/bin/env python
# coding: utf-8
# %%
import os
from os.path import join

import pickle
import pandas as pd
#from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from torchsummary import summary
from conf.localConstants import constants
from proj_io.inout import create_folder, add_forecasted_hours, add_previous_hours, filter_data, get_column_names, read_merged_files, save_columns
from proj_preproc.preproc import loadScaler
from proj_prediction.prediction import plot_forecast_hours, analyze_column

# %% Configuración Inicial
# Declaración devices...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hours_before = 24
replace_nan_value = 0
data_folder = '/ZION/AirPollutionData/Data/'
grid_size = 4
merged_specific_folder = f'{grid_size*grid_size}'
input_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/BK2/'
output_folder = os.path.join(data_folder, 'TrainingTestsPS2024')
norm_folder = join(output_folder, 'norm')
split_info_folder = join(output_folder, 'Splits')


val_perc = 0.1
test_perc = 0
epochs = 5000
batch_size = 5000
bootstrap = True
boostrap_factor = 15
boostrap_threshold = 2.9
model_name_user = 'TestPSpyt'
start_year = 2010
end_year = 2019
test_year = 2019
cur_pollutant = 'otres'
cur_station = 'MER'
forecasted_hours = 24
norm_type = 'meanzero var 1'
stations_2020 = ["UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM", "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL", "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF"]
stations = stations_2020
pollutants = "cont_otres"

# %% Creación de carpetas necesarias
folders = ['Splits', 'Parameters', 'models', 'logs', 'imgs', 'norm']
for folder in folders:
    create_folder(os.path.join(output_folder, folder))

# %% Funciones de Utilidad:

# %%
from  pytorch_proj import normalizeData, split_train_validation_and_test, apply_bootstrap, plot_forecast_hours

# %% used vars
now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = F'{model_name_user}_{cur_pollutant}_{now}'
# %% Carga y Preparación de Datos

data = read_merged_files(input_folder, start_year, end_year)

# %%

def preprocessing_data_step0(data, gen_col_csv=True,file_name_norm=None):
        """ Preprocessing data:file_name_norm:
        sorted
    else:
            formating dates, normalizing, filter contaminants, Y values building.
        return X_df, Y_df, column_x_csv, column_y_csv
            
            """

        datetimes_str = data.index.values
        datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])
        if file_name_norm:
            data_norm_df = normalizeData(data, "mean_zero", file_name_norm)

        else:
            file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  

            print("Normalizing data....")
            data_norm_df = normalizeData(data, "mean_zero", file_name_norm)
            
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(data_norm_df)

        # Here we remove all the data of other pollutants
        X_df = filter_data(data_norm_df, filter_type='single_pollutant',
                            filtered_pollutant=cur_pollutant) 

        print(X_df.columns.values)
        print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

        print(F"Building X ...")
        X_df = add_previous_hours(X_df, hours_before=hours_before)

        print("\Building Y...:Adding the forecasted hours of the pollutant as the predicted column Y...")
        Y_df = add_forecasted_hours(X_df, cur_pollutant, range(1,forecasted_hours+1))

        X_df = X_df.iloc[hours_before:,:]
        Y_df = Y_df.iloc[hours_before:,:]
        if gen_col_csv:
            column_y_csv = join(output_folder, 'Y_columns.csv')
            column_x_csv = join(output_folder, 'X_columns.csv')
            save_columns(Y_df, column_y_csv)
            save_columns(X_df, column_x_csv)

        print("Done!")

        print(F'Original {data_norm_df.shape}')
        print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
        print(F'Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')

        return X_df, Y_df, column_x_csv, column_y_csv, file_name_norm

def preprocessing_data_step1(X_df, Y_df):
        # Preprocessing data:
        # Spliting train validation sets, and bootstrapping data

        print("Splitting training and validation data by year....")
        splits_file = join(split_info_folder, F'splits_{model_name}.csv')
        train_idxs, val_idxs, test_idxs = split_train_validation_and_test(
            len(X_df), val_perc, test_perc, shuffle_ids=False, file_name=splits_file)

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

        # Managing nan values..
        print(f"Replacing nan values with {replace_nan_value}...")
        X_df_train.fillna(replace_nan_value, inplace=True)
        X_df_val.fillna(replace_nan_value, inplace=True)
        Y_df_train.fillna(replace_nan_value, inplace=True)
        Y_df_val.fillna(replace_nan_value, inplace=True)

        print(f"Train examples: {X_df_train.shape[0]}")
        print(f"Validation examples {X_df_val.shape[0]}")

        print(type(X_df_val))
        print(len(X_df_val))
        return X_df_train, Y_df_train, X_df_val, Y_df_val


# %% Preprocesssing, normalize, bootstrap and split datadata:

X_df, Y_df, column_x_csv, column_y_csv, file_name_norm = preprocessing_data_step0(data)
X_df_train, Y_df_train, X_df_val, Y_df_val = preprocessing_data_step1(X_df, Y_df)


# %%
# Convierte X_df_train Y_df.. a un tensor de PyTorch
X_df_train_tensor = torch.tensor(X_df_train.values, dtype=torch.float32)
Y_df_train_tensor = torch.tensor(Y_df_train.values, dtype=torch.float32) 
X_df_val_tensor = torch.tensor(X_df_val.values, dtype=torch.float32)
Y_df_val_tensor = torch.tensor(Y_df_val.values, dtype=torch.float32)

# Verificación de la conversión y las dimensiones
print(type(X_df_train_tensor), X_df_train_tensor.shape)
print(type(Y_df_train_tensor), Y_df_train_tensor.shape)

# %%
# Ahora puedes crear el TensorDataset
train_dataset = TensorDataset(X_df_train_tensor, Y_df_train_tensor)
val_dataset = TensorDataset(X_df_val_tensor, Y_df_val_tensor)

# Se crea Dataloaders
train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=True)


# Definición y Carga del Modelo
class MyOriginalModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyOriginalModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),
            nn.Linear(300, num_classes),
        )

    def forward(self, x):
        return self.network(x)

# Instanciación del modelo
model = MyOriginalModel(input_size=X_df_train.shape[1], num_classes=Y_df_train.shape[1])


print(f'X_df input size: {X_df_train.shape[1]}')
print(f'Y_df output size: {Y_df_train.shape[1]}')


INPUT_SIZE=X_df_train.shape[1] # 3450
NUM_CLASSES=Y_df_train.shape[1] # 720

# Configuración del dispositivo de cálculo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Mover el modelo al dispositivo
model.to(device)

# Verificar y aplicar DataParallel si hay múltiples GPUs disponibles
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs!')
    model = nn.DataParallel(model)

# %% imprimiendo summary of the model:
model.eval()  # Poner el modelo en modo de evaluación
summary(model, input_size=(INPUT_SIZE,))
model.train()  # Regresar el modelo a modo de entrenamiento


# %% Configuración de la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Asumiendo una tarea de regresión con MSE como la función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# %% Inicializa listas para almacenar las pérdidas y métricas
train_losses = []
val_losses = []
train_mse = []
val_mse = []

num_epochs = 5000
patience = 50  # Número de épocas para esperar mejora antes de detener el entrenamiento

best_val_loss = float('inf')
patience_counter = 0  # Contador para rastrear cuántas épocas han pasado sin mejora

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        mse = torch.mean((outputs - labels) ** 2)  # Calcula MSE
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_mse += mse.item() * inputs.size(0)
    
    train_losses.append(running_loss / len(train_loader.dataset))
    train_mse.append(running_mse / len(train_loader.dataset))
    
    # Validación
    model.eval()
    val_running_loss = 0.0
    val_running_mse = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mse = torch.mean((outputs - labels) ** 2)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_mse += mse.item() * inputs.size(0)
    
    val_losses.append(val_running_loss / len(val_loader.dataset))
    val_mse.append(val_running_mse / len(val_loader.dataset))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, MSE: {train_mse[-1]:.4f}, Val MSE: {val_mse[-1]:.4f}')
    
    # Comprobar si hay mejora
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0  # Resetear contador de paciencia
    else:
        patience_counter += 1  # Incrementar contador de paciencia
    
    # Detener el entrenamiento si no hay mejora después de 'patience' épocas
    if patience_counter >= patience:
        print(f'No improvement in validation loss for {patience} consecutive epochs. Stopping training.')
        break



# %% Curva de pérdida
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Pérdida en Entrenamiento')
plt.plot(val_losses, label='Pérdida en Validación')
plt.title('Curva de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Curva de error cuadrático medio
plt.subplot(1, 2, 2)
plt.plot(train_mse, label='MSE en Entrenamiento')
plt.plot(val_mse, label='MSE en Validación')
plt.title('Curva de Error Cuadrático Medio')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()

plt.show()

# %%
# Guardar el modelo
torch.save({
    'epoch': epoch,
    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_val_loss,
    'input_size': INPUT_SIZE,
    'num_classes': NUM_CLASSES,
    # Puedes agregar más metadatos según sea necesario
}, 'best_model2099.pth')

print("Model saved.")


# %%
# Evaluation test data after training:

model.eval()  # Asegura que el modelo esté en modo de evaluación

file_name_scaler = '/ZION/AirPollutionData/Data/TrainingTestsOZ/norm/TestPSpyt_otres_2024_02_16_05_19_scaler.pkl'

# file with testing data
hardcoded_input_file = '/ZION/AirPollutionData/Data/MergedDataCSV/16/BK2/2019_AllStations.csv'
data = pd.read_csv(hardcoded_input_file, index_col=0)

#datetimes_str = data.index.values
#datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

#now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
#model_name = F'{model_name_user}_{cur_pollutant}_{now}'
#file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  
#print("Normalizing data....")

##data_norm_df = normalizeData(data, "mean_zero", file_name_norm)
data_norm_df = preprocessing_data_step0(data, gen_col_csv=False,file_name_norm=file_name_scaler)
all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(data_norm_df)

# Here we remove all the data of other pollutants
##X_df = filter_data(data_norm_df, filter_type='single_pollutant',
##                   filtered_pollutant=cur_pollutant) 

##print(X_df.columns.values)

##X_df = add_previous_hours(X_df, hours_before=hours_before)

##print("\tAdding the forecasted hours of the pollutant as the predicted column Y...")
##Y_df = add_forecasted_hours(X_df, cur_pollutant, range(1,forecasted_hours+1))

##X_df = X_df.iloc[hours_before:,:]
##Y_df = Y_df.iloc[hours_before:,:]
### save_columns(Y_df, join(output_folder, 'Y_columns.csv'))
### save_columns(X_df, join(output_folder, 'X_columns.csv'))
##print("Done!")

##print(F'Original {data_norm_df.shape}')
##print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

##print(f"Replacing nan values with {replace_nan_value}...")
##X_df.fillna(replace_nan_value, inplace=True)
##Y_df.fillna(replace_nan_value, inplace=True)


##print(f"Train examples: {X_df.shape[0]}")
##print(f"Validation examples {Y_df.shape[0]}")
##print(f"Train examples: {X_df.shape[1]}")
X_df, Y_df, column_x_csv, column_y_csv, file_name_norm = preprocessing_data_step0(data, gen_col_csv=False,file_name_norm=file_name_norm)

# Ejemplo de conversión de NumPy a Tensor de PyTorch
X_test_tensor = torch.tensor(X_df.values, dtype=torch.float32)  # Asegúrate de que el tipo de datos sea correcto

# Mueve el tensor al mismo dispositivo que tu modelo
X_test_tensor = X_test_tensor.to(device)

# Asegúrate de que X_test_tensor esté en el dispositivo correcto
X_test_tensor = X_test_tensor.to(device)


with torch.no_grad():  # Desactiva el cálculo de gradientes para la inferencia
    predictions = model(X_test_tensor)

# %% Loading csv with X an Y columns
X_cols_csv = pd.read_csv(join(output_folder, 'X_columns.csv')) #'/ZION/AirPollutionData/Data/TrainingTestsOZ/Y_columns.csv') #pd.read_csv( join(path_csvs, 'X_columns.csv'))
Y_cols_csv = pd.read_csv(join(output_folder, 'Y_columns.csv')) # '/ZION/AirPollutionData/Data/TrainingTestsOZ/Y_columns.csv') # join(path_csvs, 'Y_columns.csv'))

X_cols = X_cols_csv['0'].tolist()
Y_cols = Y_cols_csv['0'].tolist()

file_name_scaler = '/ZION/AirPollutionData/Data/TrainingTestsOZ/norm/TestPSpyt_otres_2024_02_16_05_19_scaler.pkl'

scaler = loadScaler(file_name_norm)

# %% Computing predictions of test dataset:
Y_pred_ten = predictions

# Mover el tensor a la CPU
tensor_cpu = Y_pred_ten.cpu()

# Convertir a un arreglo de NumPy
Y_pred = tensor_cpu.numpy()

# %% A funciton is defined to generate custom scaler objects
from proj_prediction.prediction import compile_scaler
scaler_y = compile_scaler(scaler,Y_cols)

#%% Descale predictions and y_true, and their processing
Y_pred_descaled = scaler_y.inverse_transform(Y_pred)
y_pred_descaled_df = pd.DataFrame(Y_pred_descaled,
                                  columns=scaler_y.feature_names_in_)

print(y_pred_descaled_df.head())

y_true_df = pd.DataFrame(scaler_y.inverse_transform(Y_df),
                         columns=Y_df.columns)
print(y_true_df.head())


# %%  ********** Reading and preprocessing data *******
all_stations = [
    "UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM",
    "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL",
    "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF"
]

evaluate_stations = ["UIZ", "AJU", "ATI", "UAX"]
evaluate_hours = [1, 6, 12, 18, 24]

# %% 
params_grid = [(f'plus_{hour:02}_cont_{cur_pollutant}_{cur_station}')
               for cur_station in evaluate_stations for hour in evaluate_hours]


# In[46]:


for cur_column in params_grid:
    print(cur_column)
    # Llamar a la función con la columna deseada y argumentos adicionales
    plot_forecast_hours(
        cur_column, 
        y_true_df, 
        y_pred_descaled_df, 
        output_results_folder_img=None,
        show_grid=True, 
        x_label='Tiempo Ponosticado [horas]', 
        y_label='Nivel de Contaminante $O_3$ [ppb]',
        title_str=f'Niveles Pronosticados\nSalida:{cur_column} - Datos Prueba:2019',
        save_fig=True
    )


# TODO: traducir estos plots pasandole los Labels en Español
# %% Evaluating only a set of stations and hours
for station in evaluate_stations:
    for hour in evaluate_hours:
        try:
            cur_column = f'plus_{hour:02}_cont_otres_{station}'
            print(f'column name:{cur_column}')
            # analyze_column(cur_column)
            analyze_column(cur_column,
                           y_pred_descaled_df,
                           y_true_df,
                           test_year=test_year,
                            output_results_folder_img=None)
        except:
            continue



results_df = pd.DataFrame(columns=[
    "Columna", "Índice de correlación", "MAE", "MAPE", "MSE", "RMSE", "R2","Index of agreement"
])

for cur_column in y_pred_descaled_df.columns:
    try:
        column_results = analyze_column(cur_column,
                                        y_pred_descaled_df,
                                        y_true_df,
                                        test_year=test_year,
                                        generate_plot=False)
        #results_df = results_df.append(column_results, ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([column_results])], ignore_index=True)
    except ValueError as e:
        print(f"Error al procesar la columna {cur_column}: {e}")
        # Opcionalmente, maneja el error como prefieras aquí

print(results_df)




# %%
