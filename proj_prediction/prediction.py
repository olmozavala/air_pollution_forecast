import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
#from img_viz.common import create_folder
import os

def executeMetric(GT, NN, metric):
    not_nans = np.logical_not(np.isnan(GT))
    a = GT[not_nans].astype(np.int32)
    b = NN[not_nans].astype(np.int32)
    error = metric(a, b)
    # c = a-b
    # erroroz = np.mean((a - b)**2)
    # from scipy.stats import linregress
    # slope, intercept, r_value, p_value, std_err = linregress(a, b)
    # sop = r_value**2
    # from sklearn.metrics import r2_score
    # per = r2_score(a,b)

    return error


def compute_metrics(gt, nn, metrics, split_info, output_file, column_names=[], by_column=True):
    """
    Compute the received metrics and save the results in a csv file
    :param gt: Dataframe with the values stored by station by column
    :param nn: Result of the NN
    :param metrics:
    :param split_info:
    :param output_file:
    :param column_names:
    :param by_column:
    :return:
    """

    # Eliminate those cases where the original output is unknown

    train_ids = split_info.iloc[:,0]
    val_ids = split_info.iloc[:,1]
    test_ids = split_info.iloc[:,2]
    val_ids = val_ids.drop(pd.isna(val_ids).index.values)
    train_ids = train_ids.drop(pd.isna(train_ids).index.values)
    test_ids = test_ids.drop(pd.isna(test_ids).index.values)

    output_file = output_file.replace('.csv','')
    create_folder(os.path.dirname(output_file))

    if by_column:
        if len(column_names) == 0:
            column_names = [str(i) for i in range(len(gt[0]))]

        all_metrics = list(metrics.keys())
        all_metrics += [F"{x}_training" for x in metrics.keys()]
        all_metrics += [F"{x}_validation" for x in metrics.keys()]
        all_metrics += [F"{x}_test" for x in metrics.keys()]
        metrics_result = pd.DataFrame({col: np.zeros(len(metrics)*4) for col in column_names}, index=all_metrics)

        for metric_name, metric_f in metrics.items():
            for cur_col in column_names:
                # All errors
                GT = gt[cur_col].values
                NN = nn[cur_col].values
                error = executeMetric(GT, NN, metric_f)
                metrics_result[cur_col][metric_name] = error
                # Training errors
                if len(train_ids) > 0:
                    GT = gt[cur_col][train_ids].values
                    NN = nn[cur_col][train_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_training"] = error
                # Validation errors
                if len(val_ids) > 0:
                    GT = gt[cur_col][val_ids].values
                    NN = nn[cur_col][val_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_validation"] = error
                # Test errors
                if len(test_ids) > 0:
                    GT = gt[cur_col][test_ids].values
                    NN = nn[cur_col][test_ids].values
                    error = executeMetric(GT, NN, metric_f)
                else:
                    error = 0
                metrics_result[cur_col][F"{metric_name}_test"] = error
                # import matplotlib.pyplot as plt
                # print(metric_f(GT[0:100], NN[0:100]))
                # plt.plot(GT[0:100])
                # plt.plot(NN[0:100])
                # plt.show()

        metrics_result.to_csv(F"{output_file}.csv")
        nn_df = pd.DataFrame(nn, columns=column_names, index=gt.index)
        nn_df.to_csv(F"{output_file}_nnprediction.csv")

    return metrics_result


import matplotlib.pyplot as plt

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

    plt.savefig(join(output_folder, f'scatter_plot_{metric.lower()}.png'), dpi=300)
    plt.show()

# %% Funcion de ploteado hexbin y métricas
# Refactorizadas
from matplotlib.colors import LogNorm
import seaborn as sns
from os.path import join

def preprocess_data(cur_column, y_pred_descaled_df, y_true_df):
    # Obtener los arrays de predicciones y valores reales
    y_pred_plot = y_pred_descaled_df[cur_column].to_numpy()
    y_true_plot = y_true_df[cur_column].to_numpy()

    # Filtrar los valores válidos
    mask = ~np.isnan(y_true_plot) & ~np.isnan(y_pred_plot)
    y_true = y_true_plot[mask]
    y_prediction = y_pred_plot[mask]

    return y_true, y_prediction

def calculate_metrics(y_true, y_prediction):
    # Calcular las métricas
    mae = mean_absolute_error(y_true, y_prediction)
    mape = mean_absolute_percentage_error(y_true, y_prediction)
    mse = mean_squared_error(y_true, y_prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_prediction)

    return mae, mape, mse, rmse, r2

def analyze_column(cur_column, y_pred_descaled_df, y_true_df, test_year=None, output_results_folder_img='./', generate_plot=True):
    y_true, y_prediction = preprocess_data(cur_column, y_pred_descaled_df, y_true_df)

    # Imprimir el índice de correlación
    data = {"x": y_prediction, "y": y_true.squeeze()}
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    corr_coef = df["x"].corr(df["y"])
    print(f"Índice de correlación:                     {corr_coef:.4f}")

    mae, mape, mse, rmse, r2 = calculate_metrics(y_true, y_prediction)

    if generate_plot:
        analyze_column_plot(cur_column, y_true, y_prediction, corr_coef, mae, mape, rmse, test_year, output_results_folder_img)

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

def analyze_column_plot(cur_column, y_true, y_prediction, corr_coef, mae, mape, rmse, test_year=None, output_results_folder_img='./'):
    cur_station = cur_column.split('_')[-1]
    cur_pollutant = cur_column.split('_')[-2]

    test_str = f'{cur_station}_{cur_pollutant}_{test_year}'

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
    #ax.plot(y_true, slope * y_true + intercept, color='blue', linewidth=4, alpha=0.7, label='Ajuste lineal')

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


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define a function that creates a truncated colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Create a list of truncated colormaps
#original_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']  # Original color maps
original_cmaps = ['Blues', 'Greens', 'Greys', 'Oranges', 'Reds', 'Purples']  # Lista de mapas de colores

color_maps = [truncate_colormap(plt.get_cmap(cmap), 0.35, 1.0) for cmap in original_cmaps]  # Truncated color maps


def analyze_multi_hour_plot(station, hours, y_pred_descaled_df, y_true_df, test_year=None, output_results_folder_img='./'):
    # Definir el tamaño de los bins de hexágono
    gridsize = 30

    # Graficar el hexbin usando Matplotlib
    sns.set()
    fig, ax = plt.subplots(figsize=(9, 7))

    # Establecer los ejes X e Y con el mismo rango de unidades
    max_val = 180 
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])

    # Definir la paleta de colores
    #color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']  # Lista de mapas de colores
    #color_maps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']  # Lista de mapas de colores

    for i, hour in enumerate(hours):
        cur_column = f'plus_{hour:02}_cont_otres_{station}'

        y_true, y_pred = preprocess_data(cur_column, y_pred_descaled_df, y_true_df)
        mae, mape, mse, rmse, r2 = calculate_metrics(y_true, y_pred)

        data = {"x": y_pred, "y": y_true.squeeze()}
        df = pd.DataFrame(data)
        df.dropna(inplace=True)
        corr_coef = df["x"].corr(df["y"])

        hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, cmap=color_maps[i], norm=LogNorm(), mincnt=1, alpha=0.6)
        #cb = plt.colorbar(hb, ax=ax)
        #cb.set_label('Counts', fontsize=16)
        # Linear fit
        #ax.plot(range(0,max_val), range(0,max_val), color='red', linewidth=4, alpha=0.7, label='Pronóstico Ideal')
        slope, intercept = np.polyfit(y_true, y_pred, 1)

        # Añadir la ecuación de la recta al gráfico
        eqn = f"""Estación: {station} {test_year}
    Hora: {hour}
    Índice de correlación: {corr_coef:.4f}
    RMSE: {rmse:.2f} ppb
    Pronosticado = {slope:.2f}*Observado + {intercept:.2f}
    MAE: {mae:.2f} ppb
    MAPE: {mape:.2e} 
    N: {len(y_true)}
    """
        #ax.text(0.1, 0.75, eqn, transform=ax.transAxes, fontsize=12)

    # Agregar la línea 1 a 1 y la línea de ajuste al gráfico
    ax.plot(range(0,max_val), range(0,max_val), color='red', linewidth=4, alpha=0.7, label='Pronóstico Ideal')

    # Agregar la leyenda
    ax.legend(loc=(0.75, 0.1))

    # Etiquetas de los ejes y título del gráfico
    ax.set_xlabel(r'Nivel contaminante observado $O_3$ ppb', fontsize=18)
    ax.set_ylabel(r'Nivel contaminante pronosticado $O_3$ ppb', fontsize=18)
    plt.title(f"Estación: {station} {test_year}\n", fontsize=16)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(join(output_results_folder_img,f'multihour_hexbin_{station}.png'), dpi=300)  # Guardar la figura como PNG
    plt.show()


# scaler compiler

from copy import deepcopy

# def compile_scaler(old_scaler, new_columns):
#     # Crear una copia del objeto StandardScaler original
#     new_scaler = deepcopy(old_scaler)
    
#     # Crear listas para almacenar las nuevas medias y escalas
#     new_means = []
#     new_scales = []
    
#     # Convertir feature_names_in_ a lista
#     old_features = old_scaler.feature_names_in_.tolist()
    
#     # Iterar a través de las columnas especificadas
#     for column in new_columns:
#         # Identificar la columna original correspondiente
#         original_column = column.split("_", 2)[-1]
        
#         # Identificar el índice de la columna original en feature_names_in_
#         original_index = old_features.index(original_column)
        
#         # Añadir la media y la escala de la columna original a las nuevas listas
#         new_means.append(old_scaler.mean_[original_index])
#         new_scales.append(old_scaler.scale_[original_index])
    
#     # Actualizar los atributos mean_ y scale_ del nuevo objeto StandardScaler
#     new_scaler.mean_ = np.array(new_means)
#     new_scaler.scale_ = np.array(new_scales)
    
#     # Actualizar el atributo feature_names_in_ para que incluya solo las columnas especificadas
#     new_scaler.feature_names_in_ = new_columns
    
#     return new_scaler

from copy import deepcopy
import numpy as np

def compile_scaler(old_scaler, new_columns):
    # Create a copy of the original StandardScaler object
    new_scaler = deepcopy(old_scaler)
    
    # Create lists to store the new means, scales, and variances
    new_means = []
    new_scales = []
    new_vars = []
    new_samples = []
    
    # Convert feature_names_in_ to list
    old_features = old_scaler.feature_names_in_.tolist()
    
    # Iterate through the specified columns
    for column in new_columns:
        # Identify the corresponding original column
        original_column = column.split("_", 2)[-1]
        
        # Identify the index of the original column in feature_names_in_
        original_index = old_features.index(original_column)
        
        # Add the mean, scale, and variance of the original column to the new lists
        new_means.append(old_scaler.mean_[original_index])
        new_scales.append(old_scaler.scale_[original_index])
        new_vars.append(old_scaler.var_[original_index])
        new_samples.append(old_scaler.n_samples_seen_[original_index])
    
    # Update the mean_, scale_, and var_ attributes of the new StandardScaler object
    new_scaler.mean_ = np.array(new_means)
    new_scaler.scale_ = np.array(new_scales)
    new_scaler.var_ = np.array(new_vars)
    new_scaler.n_samples_seen_ = np.array(new_samples)
    
    # Update the feature_names_in_ attribute to only include the specified columns
    new_scaler.feature_names_in_ = new_columns
    
    # Update n_features_in_ to reflect the number of features in the new scaler
    new_scaler.n_features_in_ = len(new_columns)
    
    return new_scaler


# %% Plot de average metrics en orden alfabético
def average_metric(results_df, metric, output_results_folder_img):
    # Extraer las claves de las estaciones usando split y obtener los valores únicos con un conjunto
    station_keys = sorted(set(results_df['Columna'].apply(lambda x: x.split('_')[-1])))

    # Crear un DataFrame para almacenar las métricas promedio
    average_metrics_df = pd.DataFrame(columns=['Station', metric])

    # Iterar sobre las estaciones y calcular las métricas promedio
    for station in station_keys:
        station_df = results_df[results_df['Columna'].str.endswith(station)]
        average_metric_value = station_df[metric].mean()
        average_metrics_df = average_metrics_df.append({'Station': station, metric: average_metric_value}, ignore_index=True)

    # Calcular el promedio general de la métrica
    overall_average_metric = average_metrics_df[metric].mean()
    print(f"Promedio general de {metric} sobre todas las estaciones: {overall_average_metric:.4f}")

    # Gráfico de barras para las métricas promedio
    average_metrics_df.plot(x='Station', y=metric, kind='bar', figsize=(12, 6))
    plt.title(f'{metric} promedio por Estación\n Promedio global de {metric} sobre todas las estaciones {overall_average_metric:.4f}')
    plt.ylabel(metric)
    plt.xlabel('Estación')
    y_ticks = plt.gca().get_yticks()  # Obtener los ticks del eje y
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='dashed', alpha=0.5)
    plt.grid(True, axis='y', linestyle='dashed', alpha=0.5)  # Agregar grid en el eje y
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(join(output_results_folder_img, f'avg_metric_plot_{metric.lower()}.png'), dpi=300)
    plt.show()


# %%
def average_metric_by_hour(results_df, metric, output_results_folder_img):
    # Extraer las claves de las horas usando split y obtener los valores únicos con un conjunto
    hour_keys = sorted(set(results_df['Columna'].apply(
        lambda x: x.split('_')[1])), key=lambda x: int(x.split('plus_')[-1]))

    # Crear un DataFrame para almacenar las métricas promedio
    average_metrics_df = pd.DataFrame(columns=['Hour', metric])

    # Iterar sobre las horas y calcular las métricas promedio
    for hour in hour_keys:
        hour_df = results_df[results_df['Columna'].str.contains(hour)]
        average_metric_value = hour_df[metric].mean()
        average_metrics_df = average_metrics_df.append(
            {'Hour': hour, metric: average_metric_value}, ignore_index=True)

    # Calcular el promedio general de la métrica
    overall_average_metric = average_metrics_df[metric].mean()
    print(
        f"Promedio general de {metric} sobre todas las horas: {overall_average_metric:.4f}")

    # Gráfico de barras para las métricas promedio
    average_metrics_df.plot(x='Hour', y=metric, kind='bar', figsize=(12, 6))
    plt.title(
        f'Métrica Promedio de {metric} por hora pronosticada\n \n Promedio global de {metric} sobre todas las horas: {overall_average_metric:.4f}')
    plt.ylabel(metric)
    plt.xlabel('Hora pronosticada')

    y_ticks = plt.gca().get_yticks()  # Obtener los ticks del eje y
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='dashed', alpha=0.5)
    plt.grid(True, axis='y', linestyle='dashed',
             alpha=0.5)  # Agregar grid en el eje y

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(join(output_results_folder_img,
                f'avg_hours_plot_{metric.lower()}.png'), dpi=300)
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

def plot_max_difference(column_name, y_true_df, y_pred_descaled_df, replace_value=0):
    # Crear copias profundas de las columnas
    y_true = deepcopy(y_true_df[column_name].values)
    y_pred = deepcopy(y_pred_descaled_df[column_name].values)

    # Crear una máscara booleana donde y_true es NaN
    nan_mask = np.isnan(y_true)

    # Establecer los valores correspondientes en y_true y y_pred en replace_value
    y_true[nan_mask] = replace_value
    y_pred[nan_mask] = replace_value

    # Longitud actual de los arrays
    longitud_actual = y_true.shape[0]
    longitud_deseada = (longitud_actual // 24 + 1) * 24  # redondear hacia arriba para que sea divisible por 24

    # Agregar ceros al final de y_true y y_pred
    y_true_padded = np.pad(y_true, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)
    y_pred_padded = np.pad(y_pred, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)

    # Obtener el valor máximo de cada día en y_true e y_pred
    max_y_true = np.max(np.reshape(y_true_padded, (-1, 24)), axis=1)
    max_y_pred = np.max(np.reshape(y_pred_padded, (-1, 24)), axis=1)

    # Crear una máscara para excluir los ceros agregados por el relleno
    nonzero_mask = max_y_true != 0

    # Graficar la diferencia en valores máximos cada 24 horas
    plt.figure(figsize=[10, 5])
    plt.plot((max_y_true - max_y_pred)[nonzero_mask])
    h24_max_err_mean = np.mean((max_y_true - max_y_pred)[nonzero_mask])
    h24_max_err_std  = np.std((max_y_true - max_y_pred)[nonzero_mask])

    print(f'Error promedio al comparar valores máximos cada 24 hrs: {h24_max_err_mean}')
    print(f'Desviación estándar de diferencia en valores máximos diarios: {h24_max_err_std}')

    plt.title(f'Diferencia en valores máximos c/24 hrs {column_name}')
    plt.xlabel('Tiempo pronosticado [día]')
    plt.ylabel(r'Diferencia en niveles de $O_3$ [ppm]')
    plt.show()
    return max_y_true, max_y_pred, h24_max_err_mean, h24_max_err_std

# def plot_max_difference(column_name, y_true_df, y_pred_descaled_df, replace_value=0):
#     # Crear copias profundas de las columnas
#     y_true = deepcopy(y_true_df[column_name].values)
#     y_pred = deepcopy(y_pred_descaled_df[column_name].values)

#     # Crear una máscara booleana donde y_true es NaN
#     nan_mask = np.isnan(y_true)

#     # Establecer los valores correspondientes en y_true y y_pred en replace_value
#     y_true[nan_mask] = replace_value
#     y_pred[nan_mask] = replace_value
    
#     # Longitud actual de los arrays
#     longitud_actual = y_true.shape[0]
#     longitud_deseada = (longitud_actual // 24 + 1) * 24  # redondear hacia arriba para que sea divisible por 24

#     # Agregar ceros al final de y_true y y_pred
#     y_true = np.pad(y_true, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)
#     y_pred = np.pad(y_pred, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)

#     # Obtener el valor máximo de cada día en y_true e y_pred
#     max_y_true = np.max(np.reshape(y_true, (-1, 24)), axis=1)
#     max_y_pred = np.max(np.reshape(y_pred, (-1, 24)), axis=1)

#     # Graficar la diferencia en valores máximos cada 24 horas
#     plt.figure(figsize=[10, 5])
#     plt.plot(max_y_true - max_y_pred)
#     h24_max_err_mean = np.mean(max_y_true - max_y_pred)
#     h24_max_err_std  = np.std(max_y_true - max_y_pred)

#     print(f'Error promedio al comparar valores máximos cada 24 hrs: {h24_max_err_mean}')
#     print(f'Desviación estándar de diferencia en valores máximos diarios: {h24_max_err_std}')

#     plt.title(f'Diferencia en valores máximos c/24 hrs {column_name}')
#     plt.xlabel('Tiempo pronosticado [día]')
#     plt.ylabel(r'Diferencia en niveles de $O_3$ [ppm]')
#     plt.show()
#     return max_y_true, max_y_pred, h24_max_err_mean, h24_max_err_std


def plot_forecast_hours(column_to_plot, y_true_df, y_pred_descaled_df, output_results_folder_img=False):
    plot_this_many = 24 * 20  # Cantidad de puntos a graficar

    # Obtener las columnas para graficar
    y_true_column = y_true_df[column_to_plot]
    y_pred_column = y_pred_descaled_df[column_to_plot]

    x_plot = range(len(y_true_column))
    
    plt.figure(figsize=[10, 5])
    plt.xlabel('Tiempo pronosticado [horas]')
    plt.ylabel(r'Nivel contaminante $O_3$ [ppb]')

    # Graficar los datos originales y las predicciones
    plt.plot(x_plot[0:plot_this_many], y_true_column[0:plot_this_many],marker='.', label='Observado')
    plt.plot(x_plot[0:plot_this_many], y_pred_column[0:plot_this_many],marker='.', label='Pronosticado')
    plt.title(f'Niveles pronóstico para la columna {column_to_plot}')
    plt.legend()
    if output_results_folder_img:
        plt.savefig(join(output_results_folder_img, f'hours_plot_{column_to_plot.lower()}.png'), dpi=300)
    else:
        plt.show()

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_max_and_mean_difference(column_name, y_true_df, y_pred_descaled_df, replace_value=0, 
                                 show_plots=False, output_results_folder_img=None):
    # Crear copias profundas de las columnas
    y_true = deepcopy(y_true_df[column_name].values)
    y_pred = deepcopy(y_pred_descaled_df[column_name].values)
    
    # Crear una máscara booleana donde y_true es NaN
    nan_mask = np.isnan(y_true)
    
    # Establecer los valores correspondientes en y_true y y_pred en replace_value
    y_true[nan_mask] = replace_value
    y_pred[nan_mask] = replace_value
    
    # Longitud actual de los arrays
    longitud_actual = y_true.shape[0]
    longitud_deseada = (longitud_actual // 24 + 1) * 24
    
    # Agregar ceros al final de y_true y y_pred
    y_true_padded = np.pad(y_true, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)
    y_pred_padded = np.pad(y_pred, (0, longitud_deseada - longitud_actual), mode='constant', constant_values=0)
    
    # Obtener el valor máximo de cada día en y_true e y_pred
    max_y_true = np.max(np.reshape(y_true_padded, (-1, 24)), axis=1)
    max_y_pred = np.max(np.reshape(y_pred_padded, (-1, 24)), axis=1)
    
    # Obtener el promedio de los valores de cada día en y_true e y_pred
    mean_y_true = np.mean(np.reshape(y_true_padded, (-1, 24)), axis=1)
    mean_y_pred = np.mean(np.reshape(y_pred_padded, (-1, 24)), axis=1)
    
    # Crear una máscara para excluir los ceros agregados por el relleno
    nonzero_mask = max_y_true != 0
    
    # Calcular error medio y desviación estándar para los máximos
    h24_max_err_mean = np.mean((max_y_true - max_y_pred)[nonzero_mask])
    h24_max_err_std = np.std((max_y_true - max_y_pred)[nonzero_mask])
   
    # Calcular error medio y desviación estándar para los promedios
    h24_mean_err_mean = np.mean((mean_y_true - mean_y_pred)[nonzero_mask])
    h24_mean_err_std = np.std((mean_y_true - mean_y_pred)[nonzero_mask])

    print(column_name)
    print(f'Maximum error mean over 24 hrs: {h24_max_err_mean}')
    print(f'Maximum error std dev over 24 hrs: {h24_max_err_std}')
    print(f'Mean error mean over 24 hrs: {h24_mean_err_mean}')
    print(f'Mean error std dev over 24 hrs: {h24_mean_err_std}')
    
    if show_plots or output_results_folder_img:
        plt.figure(figsize=[10, 5])
        plt.plot((max_y_true - max_y_pred)[nonzero_mask])
        
        plt.title(f'Maximum Value Difference over 24 hrs {column_name}')
        plt.xlabel('Forecast Time [day]')
        plt.ylabel(r'Difference in \(O_3\) levels [ppm]')
        
        if output_results_folder_img:
            plt.savefig(os.path.join(output_results_folder_img, f"max_diff_{column_name}.png"))
        
        if show_plots:
            plt.show()
        
        plt.figure(figsize=[10, 5])
        plt.plot((mean_y_true - mean_y_pred)[nonzero_mask])
        
        plt.title(f'Mean Value Difference over 24 hrs {column_name}')
        plt.xlabel('Forecast Time [day]')
        plt.ylabel(r'Difference in \(O_3\) levels [ppm]')
        
        if output_results_folder_img:
            plt.savefig(os.path.join(output_results_folder_img, f"mean_diff_{column_name}.png"))
        
        if show_plots:
            plt.show()
    
    return max_y_true, max_y_pred, h24_max_err_mean, h24_max_err_std, mean_y_true, mean_y_pred, h24_mean_err_mean, h24_mean_err_std



# Define the function to calculate IMECA (Índice Metropolitano de la Calidad del Aire) in a refactored way.
def calculate_imeca(concentration, contaminant):
    """
    Calculates the IMECA (Metropolitan Air Quality Index) based on the concentration of a given contaminant.
    Parameters:
        concentration (float): The concentration of the contaminant.
        contaminant (str): The type of contaminant. It can be 'O3', 'SO2', 'NO2', 'CO', 'PM10', 'PM2.5'.
        'O3' input in [ppm]
    Returns:
        tuple: IMECA value and its category as a tuple.
    
    Example for 90[ppb] or 0.090[ppm]:

        calculate_imeca(0.090,'O3')

    """
    
    # Dictionary containing the parameters for calculating IMECA for different contaminants
    imeca_parameters = {
        'O3': [
            {'max_value': 0.070, 'multiplier': 714.29, 'add': 0, 'subtract': 0},
            {'max_value': 0.095, 'multiplier': 2041.67, 'add': 51, 'subtract': 0.071},
            {'max_value': 0.154, 'multiplier': 844.83, 'add': 101, 'subtract': 0.096},
            {'max_value': 0.204, 'multiplier': 1000, 'add': 151, 'subtract': 0.155},
            {'max_value': 0.404, 'multiplier': 497.49, 'add': 201, 'subtract': 0.205},
            {'max_value': float('inf'), 'multiplier': 1000, 'add': -104, 'subtract': 0}
        ],
        'SO2': [{'multiplier': 100 / 0.13}],
        'NO2': [{'multiplier': 100 / 0.21}],
        'CO': [{'multiplier': 100 / 11}],
        'PM10': [
            {'max_value': 40, 'multiplier': 1.25, 'add': 0, 'subtract': 0},
            {'max_value': 75, 'multiplier': 1.4412, 'add': 51, 'subtract': 41},
            {'max_value': 214, 'multiplier': 0.3551, 'add': 101, 'subtract': 76},
            {'max_value': 354, 'multiplier': 0.3525, 'add': 151, 'subtract': 215},
            {'max_value': 424, 'multiplier': 1.4348, 'add': 201, 'subtract': 355},
            {'max_value': float('inf'), 'multiplier': 1, 'add': -104, 'subtract': 0}
        ],
        'PM2.5': [
            {'max_value': 12, 'multiplier': 4.1667, 'add': 0, 'subtract': 0},
            {'max_value': 45, 'multiplier': 1.4894, 'add': 51, 'subtract': 12.1},
            {'max_value': 97.4, 'multiplier': 0.9369, 'add': 101, 'subtract': 45.1},
            {'max_value': 150.4, 'multiplier': 0.9263, 'add': 151, 'subtract': 97.5},
            {'max_value': 250.4, 'multiplier': 0.991, 'add': 201, 'subtract': 150.5},
            {'max_value': float('inf'), 'multiplier': 0.6604, 'add': 401, 'subtract': 350.5}
        ]
    }
    
    # Function to determine category based on IMECA value
    def get_category(imeca_value):
        if imeca_value < 51:
            return "Buena"
        elif 51 <= imeca_value < 101:
            return "Regular"
        elif 101 <= imeca_value < 151:
            return "Mala"
        elif 151 <= imeca_value < 201:
            return "Muy Mala"
        else:
            return "Extremadamente Mala"
    
    # Validate the contaminant type
    if contaminant not in imeca_parameters:
        return "Invalid contaminant type"
    
    # Get the appropriate calculation parameters for the given contaminant
    params_list = imeca_parameters[contaminant]
    
    # Calculate the IMECA value
    imeca_value = None
    for params in params_list:
        if 'max_value' in params and concentration <= params['max_value']:
            imeca_value = ((concentration - params.get('subtract', 0)) * params['multiplier']) + params['add']
            category = params.get('category', get_category(imeca_value))
            break
    else:
        imeca_value = ((concentration - params.get('subtract', 0)) * params['multiplier']) + params['add']
        category = params.get('category', get_category(imeca_value))
    
    return round(imeca_value), category
