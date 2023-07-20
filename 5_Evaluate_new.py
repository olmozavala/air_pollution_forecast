#%%
import sys

from viz.figure_generator import plot_input_output_data
# sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home

sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')

from datetime import date, datetime, timedelta

from viz_utils.eoa_viz import EOAImageVisualizer
from sklearn import preprocessing
from conf.localConstants import constants
from pandas import DataFrame
import pandas as pd
import time
from ai_common.constants.AI_params import *
from os.path import join
from conf.params import LocalTrainingParams

from conf.TrainingUserConfiguration import get_makeprediction_config
from io_utils.io_common import  create_folder
from AI.data_generation.utilsDataFormat import *
from ai_common.models.modelSelector import select_1d_model
from os import listdir
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from copy import deepcopy

from proj_io.inout import filter_data, add_previous_hours, add_forecasted_hours, get_column_names
from proj_preproc.preproc import loadScaler
#%%
config = get_makeprediction_config()
# *********** Reads the parameters ***********

input_file = config[ClassificationParams.input_file]
output_folder = config[ClassificationParams.output_folder]
output_imgs_folder = config[ClassificationParams.output_imgs_folder]
output_file_name = config[ClassificationParams.output_file_name]
model_weights_file = config[ClassificationParams.model_weights_file]
forecasted_hours = config[LocalTrainingParams.forecasted_hours]
pollutant = config[LocalTrainingParams.pollutants][0]

# ********** Reading and preprocessing data *******
all_stations = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                 "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                 "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]

evaluate_stations = ["UIZ", "AJU", "ATI"]
evaluate_hours = [1,6,12,18,24 ]

# Iterate over the stations
models_folder = '/ZION/AirPollutionData/Data/Training/cont_otres_UIZ/models/'
data_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/'

test_year = 2017
# %%
# cur_station is now a fixed value
cur_station = 'MER'

# No need for a loop here
# try:

# %% Para arreglar variables hardcodeadas:

training_results_folder = '/ZION/AirPollutionData/Data/TrainingTestsOZ/MultipleStations_MultiplePollutants_2010_2017'

model_name = 'WPrevHours8_MultipleStation_SinglePollutant_WITH_Bootstrap_AllForecastedHoursUpTo2017_otres_2023_07_06_17_29'

# file with testing data
hardcoded_input_file = '/ZION/AirPollutionData/Data/MergedDataCSV/16/2017_AllStations.csv'

# path to normalization data
norm_folder = join(training_results_folder,'norm')

imgs_folder = training_results_folder

path_csvs = training_results_folder

INPUT_SIZE = 2970 #X_df_train.shape[1] # <- TODO: obtener input size de X de algun lado 

hours_before = 8

print(f"""training_results_folder:{training_results_folder}
model_name:{model_name}
hardcoded_input_file:{hardcoded_input_file}
norm_folder:{norm_folder}
imgs_folder:{imgs_folder}
path_csvs:{path_csvs}
""")


print(f"""input_file:{input_file}
output_folder:{output_folder}
output_imgs_folder:{output_imgs_folder}
output_file_name:{output_file_name}
model_weights_file:{model_weights_file}
forecasted_hours:{forecasted_hours}
pollutant:{pollutant}
""")

# %% Definir o cargar los folder donde se exportaran imágenes y datos

output_results_folder = join(output_imgs_folder, model_name)
output_results_folder_img = join(output_results_folder, 'imgs')
output_results_folder_data = join(output_results_folder, 'data')

create_folder(output_results_folder)
create_folder(output_results_folder_img)
create_folder(output_results_folder_data)


# %%

model_name_user = config[TrainingParams.config_name]
print(model_name_user)
print(output_folder)

#%% Loading the best weight file on the weights folder
import glob
path_model_weights = join(f'{training_results_folder}',f'models')
print(path_model_weights)

model_weights_file = join(path_model_weights,f'{model_name}*')
weight_files = glob.glob(model_weights_file)
lista_weights =[]
for item in weight_files:
    lista_weights.append(float(item.split('/')[-1].split('-')[-1].split('.hdf5')[0]))
print(lista_weights)
model_weights_file = glob.glob(join(path_model_weights,f'{model_name}*{min(lista_weights)}*'))
model_weights_file = model_weights_file[0]
print(f'\nModel weight file to use of    {min(lista_weights)}: \n\n',model_weights_file)


#%%
# Selects the proper model file for the current station
assert len(model_weights_file) > 0
assert len(input_file) > 0

print(F"Working with: {model_weights_file} and input: {input_file}")


#%% data test file
input_file = hardcoded_input_file
data = pd.read_csv(input_file, index_col=0)

#%% rehacer los datos del dataset test como en los hotvectors usados durante el training.

datetimes_str = data.index.values
datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

# %% -------- Normalizing data
# loading of original scaler object
file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  
scaler = loadScaler(file_name_norm)


#%%
print("Normalizing data....")
data_norm_np = scaler.transform(data)
data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)

print(data_norm_df)
#%% Continuamos para obtener hotvector

# %% ====== Getting all the orignal columns by type
all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(data_norm_df)

# %% ====== Remove columns for other pollutants


cur_pollutant = 'otres'
X_df = filter_data(data_norm_df, filter_type='single_pollutant',
                   filtered_pollutant=cur_pollutant) 

print(X_df.columns.values)
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

# %% ====== Adding the previous hours of the pollutants as extra columns (all contaminants)
print(F"Building X and Y ....")
X_df = add_previous_hours(X_df, hours_before)

# %% ====== Adding the forecasted hours of the pollutants as the predicted column Y (specific contaminant)
print("\tAdding the forecasted hours of the pollutant as the predicted column Y...")
Y_df = add_forecasted_hours(X_df, cur_pollutant, range(1,forecasted_hours+1))

# %% Remove the first hours because Print the final shape of X and Y
X_df = X_df.iloc[hours_before:,:]
Y_df = Y_df.iloc[hours_before:,:]
print("Done!")

print(F'Original {data_norm_df.shape}')
print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
print(F'Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')


# %%
print("Removing time index...")
X_df.reset_index(drop=True, inplace=True)
Y_df.reset_index(drop=True, inplace=True)

#%% Replace all the nan values with another value
replace_value = 0
print(f"Replacing nan values with {replace_value}...")
X_df.fillna(replace_value, inplace=True)
Y_df.fillna(replace_value, inplace=True)

 
# %% Visualize input and outputs
print("Visualizing input and outputs...")

plot_input_output_data(X_df, Y_df, cur_station, cur_pollutant, 
                       output_results_folder_img, model_name) # previously used imgs_folder


# %% *********** Chooses the proper model and weights ***********

# %% Loading csv with X an Y columns
X_cols_csv = pd.read_csv( join(path_csvs, 'X_columns.csv'))
Y_cols_csv = pd.read_csv( join(path_csvs, 'Y_columns.csv'))

X_cols = X_cols_csv['0'].tolist()
Y_cols = Y_cols_csv['0'].tolist()


# %% ******************* Selecting the model **********************
# TODO: corregir hardcoded input_size
config[ModelParams.INPUT_SIZE] = INPUT_SIZE
config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = len(Y_cols)

print('Reading model ....')
model = select_1d_model(config)

# *********** Reads the weights***********
print('Reading weights ....')
model.load_weights(model_weights_file)


# %% Calculo de predicciones de el dataset de test:
Y_pred = model.predict(X_df.values)

# %% A funciton is defined to generate custom scaler objects

# TODO: Move a proj_module
def compile_scaler(old_scaler, new_columns):
    # Crear una copia del objeto StandardScaler original
    new_scaler = deepcopy(old_scaler)
    
    # Crear listas para almacenar las nuevas medias y escalas
    new_means = []
    new_scales = []
    
    # Convertir feature_names_in_ a lista
    old_features = old_scaler.feature_names_in_.tolist()
    
    # Iterar a través de las columnas especificadas
    for column in new_columns:
        # Identificar la columna original correspondiente
        original_column = column.split("_", 2)[-1]
        
        # Identificar el índice de la columna original en feature_names_in_
        original_index = old_features.index(original_column)
        
        # Añadir la media y la escala de la columna original a las nuevas listas
        new_means.append(old_scaler.mean_[original_index])
        new_scales.append(old_scaler.scale_[original_index])
    
    # Actualizar los atributos mean_ y scale_ del nuevo objeto StandardScaler
    new_scaler.mean_ = np.array(new_means)
    new_scaler.scale_ = np.array(new_scales)
    
    # Actualizar el atributo feature_names_in_ para que incluya solo las columnas especificadas
    new_scaler.feature_names_in_ = new_columns
    
    return new_scaler


scaler_y = compile_scaler(scaler,Y_cols)


#%% Desescalar las predicciones
Y_pred_descaled = scaler_y.inverse_transform(Y_pred)

# Convertir Y_pred_descaled en un DataFrame de pandas
y_pred_descaled_df = pd.DataFrame(Y_pred_descaled, columns=scaler_y.feature_names_in_)
# Verificar el DataFrame
print(y_pred_descaled_df.head())


# %% Descaling Y_df to get y_true_df
y_true_df = pd.DataFrame(scaler_y.inverse_transform(Y_df), columns=Y_df.columns)
# Verificar el DataFrame
print(y_true_df.head())


# %% Funcion de ploteado hexbin y métricas
def analyze_column_plot(cur_column):
    cur_station = cur_column.split('_')[-1]
    # Obtener los arrays de predicciones y valores reales
    y_pred_plot = y_pred_descaled_df[cur_column].to_numpy()
    y_true_plot = y_true_df[cur_column].to_numpy()

    test_str = f'{cur_station}_{cur_pollutant}_{test_year}'

    # Imprimir el índice de correlación
    data = {"x": y_pred_plot, "y": y_true_plot.squeeze()}
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    corr_coef = df["x"].corr(df["y"])
    print(f"Índice de correlación:                     {corr_coef:.4f}")

    # Filtrar los valores válidos
    mask = ~np.isnan(y_true_plot) & ~np.isnan(y_pred_plot)
    y_true = y_true_plot[mask]
    y_prediction = y_pred_plot[mask]

    # Calcular las métricas
    mae = mean_absolute_error(y_true, y_prediction)
    mape = mean_absolute_percentage_error(y_true, y_prediction)
    mse = mean_squared_error(y_true, y_prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_prediction)

    # Definir el tamaño de los bins de hexágono
    gridsize = 30

    # Graficar el hexbin usando Matplotlib
    sns.set()
    fig, ax = plt.subplots(figsize=(9, 7))

    # Establecer los ejes X e Y con el mismo rango de unidades
    max_val = 180 
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])

    hb = ax.hexbin(y_true, y_prediction, gridsize=gridsize, cmap="YlGnBu", norm=LogNorm(), mincnt=1)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Counts', fontsize=16)

    # Agregar la línea 1 a 1 y la línea de ajuste al gráfico
    ax.plot(range(0,max_val), range(0,max_val), color='red', linewidth=4, alpha=0.7, label='Pronóstico Ideal')
    slope, intercept = np.polyfit(y_true, y_prediction, 1)
    ax.plot(y_true, slope * y_true + intercept, color='blue', linewidth=4, alpha=0.7, label='Ajuste lineal')

    # Etiquetas de los ejes y título del gráfico
    ax.set_xlabel(r'Nivel contaminante observado $O_3$ ppb', fontsize=18)
    ax.set_ylabel(r'Nivel contaminante pronosticado $O_3$ ppb', fontsize=18)
    plt.title(f"Estación: {cur_station} {test_year}\n", fontsize=16)

    # Añadir la ecuación de la recta al gráfico
    eqn = f"""Estación: {cur_station} {test_year}
    Índice de correlación: {corr_coef:.4f}
    RMSE: {rmse:.2f} ppb
    Pronosticado = {slope:.2f}*Observado + {intercept:.2f}
    MAE: {mae:.2f} ppb
    MAPE: {mape:.2e} 
    N: {len(y_true)}
    """
    ax.text(0.1, 0.75, eqn, transform=ax.transAxes, fontsize=12)

    # Agregar la leyenda
    ax.legend(loc=(0.75, 0.1))

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(join(output_results_folder_img,f'hexbin_{cur_column}.png'), dpi=300)  # Guardar la figura como PNG
    plt.show()



#%%
def analyze_column(cur_column, generate_plot=True):
    # Obtener los arrays de predicciones y valores reales
    y_pred_plot = y_pred_descaled_df[cur_column].to_numpy()
    y_true_plot = y_true_df[cur_column].to_numpy()

    # Imprimir el índice de correlación
    data = {"x": y_pred_plot, "y": y_true_plot.squeeze()}
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    corr_coef = df["x"].corr(df["y"])
    print(f"Índice de correlación:                     {corr_coef:.4f}")

    # Filtrar los valores válidos
    mask = ~np.isnan(y_true_plot) & ~np.isnan(y_pred_plot)
    y_true = y_true_plot[mask]
    y_prediction = y_pred_plot[mask]

    # Calcular las métricas
    mae = mean_absolute_error(y_true, y_prediction)
    mape = mean_absolute_percentage_error(y_true, y_prediction)
    mse = mean_squared_error(y_true, y_prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_prediction)

    if generate_plot:
        # No incluir el código de trazado en esta versión
        analyze_column_plot(cur_column)

    # Retornar las métricas en un diccionario
    results = {
        "Columna": cur_column,
        "Índice de correlación": corr_coef,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    return results


# %% Evaluate only a set of stations and hours
for station in evaluate_stations:
    for hour in evaluate_hours:
        cur_column = f'plus_{hour:02}_cont_otres_{station}'
        print(f'column name:{cur_column}')
        analyze_column(cur_column)


# %% #Crear DataFrame y guardar los resultados en csv
results_df = pd.DataFrame(columns=["Columna", "Índice de correlación", "MAE", "MAPE", "MSE", "RMSE", "R2"])

# Iterar sobre las columnas
for cur_column in y_pred_descaled_df.columns:
    print(cur_column)
    column_results = analyze_column(cur_column, generate_plot=False)
    results_df = results_df.append(column_results, ignore_index=True)

# Imprimir el DataFrame de resultados
print(results_df)
results_df.to_csv(join(output_results_folder_data,'results_df.csv'), index=False)

# %% Scatter plot for the given metrics
def scatter_plot_by_column(df, metric, output_folder):
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df[metric])
    plt.xlabel('Columna')
    plt.ylabel(metric)
    plt.title(f'{metric} por columna')

    # Ajustar los ticks y etiquetas del eje x
    x_ticks = df.index[::30]  # Obtener cada 30º índice
    x_labels = df['Columna'][::30]  # Obtener cada 30º nombre de columna
    plt.xticks(x_ticks, x_labels, rotation=90)

    # Agregar líneas de grid vertical y horizontalmente
    for x in x_ticks:
        plt.axvline(x, color='gray', linestyle='dashed', alpha=0.5)
    plt.grid(True, axis='x', linestyle='dashed', alpha=0.5)  # Agregar grid en el eje x

    y_ticks = plt.gca().get_yticks()  # Obtener los ticks del eje y
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='dashed', alpha=0.5)
    plt.grid(True, axis='y', linestyle='dashed', alpha=0.5)  # Agregar grid en el eje y

    plt.savefig(join(output_folder, f'{metric.lower()}_scatter_plot.png'), dpi=300)
    plt.show()


scatter_plot_by_column(results_df, 'Índice de correlación', output_results_folder_img)
scatter_plot_by_column(results_df, 'RMSE', output_results_folder_img)
scatter_plot_by_column(results_df, 'MAE', output_results_folder_img)