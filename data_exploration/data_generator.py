# ** file data_generator.py

# *** funciones de bases de datos:

# Obtiene datos de una o mas estaciones para un contaminante en
# especifico en una fecha dada.

# pollutant_by_stations(start_date, end_date, stations [], pollutant) →
# arreglo de [stations, datos]

# Maximos diarios de una o mas estaciones para un contaminante

# daily_max_pollutant_by_stations(start_date, end_date, stations [],
# pollutant) → arreglo de [stations, dias]

# Dias arriba de algun umbral para alguna de las estaciones (considera
# todas al mismo tiempo)

# date_above_threshold(start_date, end_date, threshold, pollutant) →
# arreglo de [dates, stations, and value] → ejemplo de un renglon
# [2010-01-02, ['estacion1', ‘estacion 2'…], [val1, val2, ….] ]

# Campo promedio para los siguientes X horas. Promediar todo el campo de
# inicio y proveer el promedio

# average_meteo(start_date, hours, field (temp, u, v, etc)) → arreglo
# con [promedios (size hours)]

#####################################################################

# %% pollutant_by_stations
import sys
import os
script_dir = os.path.dirname(__file__) # Para importar el path del proyecto
project_root = os.path.join(script_dir, '..')  # 'os.path.join' ruta subiendo un nivel
sys.path.append(project_root)
import pandas as pd
from datetime import datetime
from collections import defaultdict
from db.queries import getPollutantFromDateRange
from db.sqlCont import getPostgresConn

def pollutant_by_stations(start_date: str, end_date: str, stations: list, pollutant: str) -> pd.DataFrame:
    """
    Get pollution data for specific stations and pollutant within a date range.

    Parameters:
        start_date (str): The start date in the format "YYYY-MM-DD HH:MM".
        end_date (str): The end date in the format "YYYY-MM-DD HH:MM".
        stations (list): A list of station codes.
        pollutant (str): The pollutant to query, e.g., "cont_otres" or "cont_pmdoscinco".

    Returns:
        pd.DataFrame: A DataFrame where each column represents a station and each row represents a timestamp and the corresponding value.
    """
    conn = getPostgresConn()
    res_data = getPollutantFromDateRange(conn, pollutant, datetime.strptime(start_date, '%Y-%m-%d %H:%M'), 
                                         datetime.strptime(end_date, '%Y-%m-%d %H:%M'), stations)
    conn.close()

    # Create a dictionary to hold data per station
    data_dict = defaultdict(lambda: pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='H')))
    for timestamp, value, station in res_data:
        if station in stations:
            data_dict[station][timestamp] = value

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data_dict)
    return df


# Example usage
# start_date = "2023-12-01 00:00"
# end_date = "2023-12-14 00:00"
# stations = ['UAX', 'MER', 'XAL']
# pollutant = 'cont_otres'

# data = pollutant_by_stations(start_date, end_date, stations, pollutant)
# print(data)

#########################################################################
# %% daily_max_pollutant_by_stations

def daily_max_pollutant_by_stations(start_date: str, end_date: str, stations: list, pollutant: str) -> pd.DataFrame:
    """
    Get the daily maximum pollution data for specific stations and pollutant within a date range, returning the exact date and value.

    Parameters:
        start_date (str): The start date in the format "YYYY-MM-DD HH:MM".
        end_date (str): The end date in the format "YYYY-MM-DD HH:MM".
        stations (list): A list of station codes.
        pollutant (str): The specific pollutant to query, e.g., "cont_otres" or "cont_pmdoscinco".

    Returns:
        pd.DataFrame: A DataFrame where each row represents the maximum value recorded for that day at a station,
                       including the date of the maximum, the station code, and the value.
    """
    # First, get the complete data set using the previously defined function
    pollution_data = pollutant_by_stations(start_date, end_date, stations, pollutant)

    # Prepare a DataFrame to collect the results
    results = []

    # Calculate the daily maximum for each station
    for station in stations:
        if station in pollution_data.columns:  # Check if the station has data
            daily_data = pollution_data[station].resample('D').max()
            for date, value in daily_data.iteritems():
                if pd.notna(value):  # Ensure we only include days where data exists
                    max_time = pollution_data[station][pollution_data[station].index.date == date.date()].idxmax()
                    results.append({'date': max_time, 'station': station, 'value': value})

    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


# Example usage

# start_date = "2022-12-01 00:00"
# end_date = "2023-12-14 00:00"
# stations = ['UAX', 'MER', 'XAL']
# pollutant = 'cont_otres'

# data = daily_max_pollutant_by_stations(start_date, end_date, stations, pollutant)
# print(data)

#########################################################################
# %% dates_above_threshold()
def dates_above_threshold(df, threshold=90.0):
    """
    Replace values below a specified threshold with NaN in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with pollution data.
        threshold (float): Threshold below which values will be replaced with NaN.

    Returns:
        pd.DataFrame: The modified DataFrame with values below the threshold replaced by NaN.
    """
    df_modified = df.where(df >= threshold)
    # Limpiar el DataFrame eliminando filas donde todas las estaciones son NaN
    df_cleaned = df_modified.dropna(how='all')

    return df_cleaned

## Ejemplo de uso
# start_date = "2024-05-9 00:00"
# end_date = "2024-05-14 00:00"
# #stations = ['UAX', 'MER', 'XAL']
# all_stations = [
#     "UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM",
#     "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL",
#     "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF"
# ]
# pollutant = 'cont_otres'

# df = pollutant_by_stations(start_date, end_date, all_stations, pollutant)
# df_above_th = dates_above_threshold(df, 145.0)

# print(df_above_th)


#########################################################################
# %% average_meteo function
#########################################################################

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from os.path import join
from proj_io.inout import get_month_folder_esp
from proj_preproc.wrf import crop_variables_xr, subsampleData

wrf_input_folder = "/ServerData/WRF_2017_Kraken"
bbox = [18.75, 20, -99.75, -98.5]
grid_size_wrf = 4  # 4 for 4x4

def meteo_per_hour(start_date, hours, field):
    """
    Get the hourly values for a specified meteorological field from a starting date over a specified number of hours,
    correcting for GMT.

    Parameters:
        start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
        hours (int): The number of hours to retrieve values for from the start date.
        field (str): The meteorological field to retrieve (e.g., 'T2', 'U10', 'V10').

    Returns:
        pd.DataFrame: A DataFrame with the values of the specified field over the given number of hours.
    """
    # Parse the starting date
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    gmt_correction = 6  # Correcting to match UTC time used by WRF data
    start_datetime_wrf = start_datetime + timedelta(hours=gmt_correction)
    end_datetime_wrf = start_datetime_wrf + timedelta(hours=hours)
    
    all_meteo_data = []
    file_names = []
    current_datetime_wrf = start_datetime_wrf

    while current_datetime_wrf < end_datetime_wrf:
        year = current_datetime_wrf.year
        month = current_datetime_wrf.month
        cur_month = get_month_folder_esp(month)  # Get folder in Spanish names spec
        wrf_file_name = f"wrfout_d01_{current_datetime_wrf.strftime('%Y-%m-%d')}_00.nc"
        wrf_file = join(join(wrf_input_folder, str(current_datetime_wrf.year), cur_month), wrf_file_name)
        
        days_before_wrf = 0  # Variable to check several nc's, in case the last is not found
        wrf_times_to_load = range(24)
        meteo_var_names = [field]
        
        while not os.path.exists(wrf_file):
            print(f"ERROR: File {wrf_file} does not exist")
            days_before_wrf += 1
            wrf_run_day = current_datetime_wrf - timedelta(days=days_before_wrf)
            wrf_file_name = f"wrfout_d01_{wrf_run_day.strftime('%Y-%m-%d')}_00.nc"
            wrf_file = join(join(wrf_input_folder, str(wrf_run_day.year), cur_month), wrf_file_name)
            wrf_times_to_load = np.array(list(wrf_times_to_load)) + 24  # Modify reading 24 'shifted' hours

        #print(f"Working with wrf file: {wrf_file}")
        
        cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
        cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=wrf_times_to_load)
        
        # Subsampling the data
        subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
        wrf_time_format = '%Y-%m-%d_%H:%M:%S'
        wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values[wrf_times_to_load]])
        
        if current_datetime_wrf == start_datetime_wrf:
            first_time_idx = np.where(wrf_dates >= current_datetime_wrf)[0][0]
        else:
            first_time_idx = 0

        hours_to_load = min(hours, len(wrf_dates) - first_time_idx)
        for wrf_hour_index in range(first_time_idx, first_time_idx + hours_to_load):
            cur_var_np = subsampled_xr_ds[field].values[wrf_hour_index, :, :]
            all_meteo_data.append(cur_var_np.flatten())
            file_names.append(wrf_file_name)

        current_datetime_wrf += timedelta(hours=int(hours_to_load))
        hours -= hours_to_load

    meteo_df = pd.DataFrame(all_meteo_data, index=pd.date_range(start=start_datetime_wrf, periods=len(all_meteo_data), freq='H'))
    meteo_df['wrf_file_name'] = file_names
    
    return meteo_df

# Función para calcular la velocidad del viento (WSP) y la dirección del viento (WDR) desde U10 y V10
def calculate_wind_speed_direction(u10_df, v10_df):
    # Seleccionar solo columnas numéricas
    u10_numeric = u10_df.select_dtypes(include=[np.number])
    v10_numeric = v10_df.select_dtypes(include=[np.number])
    
    wsp_df = np.sqrt(u10_numeric**2 + v10_numeric**2)
    wdr_df = (180 / np.pi * np.arctan2(u10_numeric, v10_numeric) + 360) % 360
    return wsp_df, wdr_df

# Función para calcular la media solo de columnas numéricas
def average_meteo(start_date, hours, field):
    """
    Calculate the average of the grid values for each hour for a specified meteorological field 
    from a starting date over a specified number of hours.

    Parameters:
        start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
        hours (int): The number of hours to average over from the start date.
        field (str): The meteorological field to average (e.g., 'T2', 'U10', 'V10').

    Returns:
        pd.Series: A Series with the average values of the specified field for each hour.
    """
    if field == 'WIND':
        u10_df = meteo_per_hour(start_date, hours, 'U10')
        v10_df = meteo_per_hour(start_date, hours, 'V10')
        wsp_df, wdr_df = calculate_wind_speed_direction(u10_df, v10_df)
        # Seleccionar solo columnas numéricas
        numeric_wsp_df = wsp_df.select_dtypes(include=[np.number])
        numeric_wdr_df = wdr_df.select_dtypes(include=[np.number])
        return numeric_wsp_df.mean(axis=1), numeric_wdr_df.mean(axis=1)
    else:
        meteo_df = meteo_per_hour(start_date, hours, field)
        # Seleccionar solo columnas numéricas
        numeric_df = meteo_df.select_dtypes(include=[np.number])
        return numeric_df.mean(axis=1)

# Ejemplo de uso
# start_date = "2022-05-03 00:00"
# hours = 48
# field = 'WIND'  # Example meteorological field, e.g., temperature

# if field == 'WIND':
#     wsp_average, wdr_average = average_meteo(start_date, hours, field)
#     print("Wind Speed Average:\n", wsp_average)
#     print("Wind Direction Average:\n", wdr_average)
# else:
#     average_values = average_meteo(start_date, hours, field)
#     print(average_values)


# import os
# import numpy as np
# import pandas as pd
# import xarray as xr
# from datetime import datetime, timedelta
# from os.path import join
# from proj_io.inout import get_month_folder_esp
# from proj_preproc.wrf import crop_variables_xr, subsampleData

# wrf_input_folder = "/ServerData/WRF_2017_Kraken"
# bbox = [18.75, 20, -99.75, -98.5]
# grid_size_wrf = 4  # 4 for 4x4

# def meteo_per_hour(start_date, hours, field):
#     """
#     Get the hourly values for a specified meteorological field from a starting date over a specified number of hours,
#     correcting for GMT.

#     Parameters:
#         start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
#         hours (int): The number of hours to retrieve values for from the start date.
#         field (str): The meteorological field to retrieve (e.g., 'T2', 'U10', 'V10').

#     Returns:
#         pd.DataFrame: A DataFrame with the values of the specified field over the given number of hours.
#     """
#     # Parse the starting date
#     start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
#     gmt_correction = 6  # Correcting to match UTC time used by WRF data
#     start_datetime_wrf = start_datetime + timedelta(hours=gmt_correction)
#     end_datetime_wrf = start_datetime_wrf + timedelta(hours=hours)
    
#     all_meteo_data = []
#     file_names = []
#     current_datetime_wrf = start_datetime_wrf

#     while current_datetime_wrf < end_datetime_wrf:
#         year = current_datetime_wrf.year
#         month = current_datetime_wrf.month
#         cur_month = get_month_folder_esp(month)  # Get folder in Spanish names spec
#         wrf_file_name = f"wrfout_d01_{current_datetime_wrf.strftime('%Y-%m-%d')}_00.nc"
#         wrf_file = join(join(wrf_input_folder, str(current_datetime_wrf.year), cur_month), wrf_file_name)
        
#         days_before_wrf = 0  # Variable to check several nc's, in case the last is not found
#         wrf_times_to_load = range(24)
#         meteo_var_names = [field]
        
#         while not os.path.exists(wrf_file):
#             print(f"ERROR: File {wrf_file} does not exist")
#             days_before_wrf += 1
#             wrf_run_day = current_datetime_wrf - timedelta(days=days_before_wrf)
#             wrf_file_name = f"wrfout_d01_{wrf_run_day.strftime('%Y-%m-%d')}_00.nc"
#             wrf_file = join(join(wrf_input_folder, str(wrf_run_day.year), cur_month), wrf_file_name)
#             wrf_times_to_load = np.array(list(wrf_times_to_load)) + 24  # Modify reading 24 'shifted' hours

#         #print(f"Working with wrf file: {wrf_file}")
        
#         cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
#         cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=wrf_times_to_load)
        
#         # Subsampling the data
#         subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
#         wrf_time_format = '%Y-%m-%d_%H:%M:%S'
#         wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values[wrf_times_to_load]])
        
#         if current_datetime_wrf == start_datetime_wrf:
#             first_time_idx = np.where(wrf_dates >= current_datetime_wrf)[0][0]
#         else:
#             first_time_idx = 0

#         hours_to_load = min(hours, len(wrf_dates) - first_time_idx)
#         for wrf_hour_index in range(first_time_idx, first_time_idx + hours_to_load):
#             cur_var_np = subsampled_xr_ds[field].values[wrf_hour_index, :, :]
#             all_meteo_data.append(cur_var_np.flatten())
#             file_names.append(wrf_file_name)

#         current_datetime_wrf += timedelta(hours=int(hours_to_load))
#         hours -= hours_to_load

#     meteo_df = pd.DataFrame(all_meteo_data, index=pd.date_range(start=start_datetime_wrf, periods=len(all_meteo_data), freq='H'))
#     meteo_df['wrf_file_name'] = file_names
    
#     return meteo_df

# # Función para calcular la velocidad del viento (WSP) y la dirección del viento (WDR) desde U10 y V10
# def calculate_wind_speed_direction(u10_df, v10_df):
#     wsp_df = np.sqrt(u10_df**2 + v10_df**2)
#     wdr_df = (180 / np.pi * np.arctan2(u10_df, v10_df) + 360) % 360
#     return wsp_df, wdr_df

# # Función para calcular la media solo de columnas numéricas
# def average_meteo(start_date, hours, field):
#     """
#     Calculate the average of the grid values for each hour for a specified meteorological field 
#     from a starting date over a specified number of hours.

#     Parameters:
#         start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
#         hours (int): The number of hours to average over from the start date.
#         field (str): The meteorological field to average (e.g., 'T2', 'U10', 'V10').

#     Returns:
#         pd.Series: A Series with the average values of the specified field for each hour.
#     """
#     if field == 'WIND':
#         u10_df = meteo_per_hour(start_date, hours, 'U10')
#         v10_df = meteo_per_hour(start_date, hours, 'V10')
#         wsp_df, wdr_df = calculate_wind_speed_direction(u10_df, v10_df)
#         # Seleccionar solo columnas numéricas
#         numeric_wsp_df = wsp_df.select_dtypes(include=[np.number])
#         numeric_wdr_df = wdr_df.select_dtypes(include=[np.number])
#         return numeric_wsp_df.mean(axis=1), numeric_wdr_df.mean(axis=1)
#     else:
#         meteo_df = meteo_per_hour(start_date, hours, field)
#         # Seleccionar solo columnas numéricas
#         numeric_df = meteo_df.select_dtypes(include=[np.number])
#         return numeric_df.mean(axis=1)

# Ejemplo de uso
#start_date = "2022-05-03 00:00"
#hours = 48
#field = 'WIND'  # Example meteorological field, e.g., temperature

#if field == 'WIND':
#    wsp_average, wdr_average = average_meteo(start_date, hours, field)
#    print("Wind Speed Average:\n", wsp_average)
#    print("Wind Direction Average:\n", wdr_average)
#else:
#    average_values = average_meteo(start_date, hours, field)
#    print(average_values)

# import os
# import numpy as np
# import pandas as pd
# import xarray as xr
# from datetime import datetime, timedelta
# from os.path import join
# from proj_io.inout import get_month_folder_esp
# from proj_preproc.wrf import crop_variables_xr, subsampleData
# import matplotlib.pyplot as plt

# wrf_input_folder = "/ServerData/WRF_2017_Kraken"
# bbox = [18.75, 20, -99.75, -98.5]
# grid_size_wrf = 4  # 4 for 4x4

# def meteo_per_hour(start_date, hours, field):
#     """
#     Get the hourly values for a specified meteorological field from a starting date over a specified number of hours,
#     correcting for GMT.

#     Parameters:
#         start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
#         hours (int): The number of hours to retrieve values for from the start date.
#         field (str): The meteorological field to retrieve (e.g., 'T2', 'U10', 'V10').

#     Returns:
#         pd.DataFrame: A DataFrame with the values of the specified field over the given number of hours.
#     """
#     # Parse the starting date
#     start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
#     gmt_correction = 6  # Correcting to match UTC time used by WRF data
#     start_datetime_wrf = start_datetime + timedelta(hours=gmt_correction)
#     end_datetime_wrf = start_datetime_wrf + timedelta(hours=hours)
    
#     all_meteo_data = []
#     file_names = []
#     current_datetime_wrf = start_datetime_wrf
#     hours_processed_per_day = 0

#     while current_datetime_wrf < end_datetime_wrf:
#         year = current_datetime_wrf.year
#         month = current_datetime_wrf.month
#         cur_month = get_month_folder_esp(month)  # Get folder in Spanish names spec
#         wrf_file_name = f"wrfout_d01_{current_datetime_wrf.strftime('%Y-%m-%d')}_00.nc"
#         wrf_file = join(join(wrf_input_folder, str(current_datetime_wrf.year), cur_month), wrf_file_name)
        
#         days_before_wrf = 0  # Variable to check several nc's, in case the last is not found
#         wrf_times_to_load = range(24)
#         meteo_var_names = [field]
        
#         while not os.path.exists(wrf_file):
#             print(f"ERROR: File {wrf_file} does not exist")
#             days_before_wrf += 1
#             wrf_run_day = current_datetime_wrf - timedelta(days=days_before_wrf)
#             wrf_file_name = f"wrfout_d01_{wrf_run_day.strftime('%Y-%m-%d')}_00.nc"
#             wrf_file = join(join(wrf_input_folder, str(wrf_run_day.year), cur_month), wrf_file_name)
#             wrf_times_to_load = np.array(list(wrf_times_to_load)) + 24  # Modify reading 24 'shifted' hours

#         #print(f"Working with wrf file: {wrf_file}")
        
#         cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
#         cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=wrf_times_to_load)
        
#         # Subsampling the data
#         subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
#         wrf_time_format = '%Y-%m-%d_%H:%M:%S'
#         wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values[wrf_times_to_load]])
        
#         if current_datetime_wrf == start_datetime_wrf:
#             first_time_idx = np.where(wrf_dates >= current_datetime_wrf)[0][0]
#         else:
#             first_time_idx = 0

#         hours_to_load = min(hours, len(wrf_dates) - first_time_idx)
#         for wrf_hour_index in range(first_time_idx, first_time_idx + hours_to_load):
#             cur_var_np = subsampled_xr_ds[field].values[wrf_hour_index, :, :]
#             all_meteo_data.append(cur_var_np.flatten())
#             file_names.append(wrf_file_name)

#         current_datetime_wrf += timedelta(hours=int(hours_to_load))
#         hours -= hours_to_load

#     meteo_df = pd.DataFrame(all_meteo_data, index=pd.date_range(start=start_datetime_wrf, periods=len(all_meteo_data), freq='H'))
#     meteo_df['wrf_file_name'] = file_names
    
#     return meteo_df

# # def average_meteo(start_date, hours, field):
# #     """
# #     Calculate the average of the grid values for each hour for a specified meteorological field 
# #     from a starting date over a specified number of hours.

# #     Parameters:
# #         start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
# #         hours (int): The number of hours to average over from the start date.
# #         field (str): The meteorological field to average (e.g., 'T2', 'U10', 'V10').

# #     Returns:
# #         pd.Series: A Series with the average values of the specified field for each hour.
# #     """
# #     meteo_df = meteo_per_hour(start_date, hours, field)
# #     return meteo_df.mean(axis=1)


# # Función para calcular la media solo de columnas numéricas
# def average_meteo(start_date, hours, field):
#     """
#     Calculate the average of the grid values for each hour for a specified meteorological field 
#     from a starting date over a specified number of hours.

#     Parameters:
#         start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
#         hours (int): The number of hours to average over from the start date.
#         field (str): The meteorological field to average (e.g., 'T2', 'U10', 'V10').

#     Returns:
#         pd.Series: A Series with the average values of the specified field for each hour.
#     """
#     meteo_df = meteo_per_hour(start_date, hours, field)
#     # Seleccionar solo columnas numéricas
#     numeric_df = meteo_df.select_dtypes(include=[np.number])
#     return numeric_df.mean(axis=1)

def plot_average_values(average_values, title='Valores Promedio de Campo Meteorológico', ylabel='Valor Promedio'):
    """
    Función para graficar los valores promedio.

    Parameters:
        average_values (pd.Series): Serie con los valores promedio de un campo meteorológico.
        title (str): Título del gráfico.
        ylabel (str): Etiqueta del eje y.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(average_values.index, average_values.values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Fecha y Hora')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso

# start_date = "2023-05-09 00:00"
# hours = 24*10
# field = 'T2'  # Example meteorological field, e.g., temperature
# average_values = average_meteo(start_date, hours, field)
# print(average_values)

# Graficar los valores promedio
# plot_average_values(average_values)

# %%
