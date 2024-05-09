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

# %%
import sys
import os

# Obtiene el directorio donde se encuentra el script actualmente ejecutándose
script_dir = os.path.dirname(__file__)  # Esto devuelve el directorio de 'your_script.py'

# Añade el directorio raíz del proyecto al path de Python
# Suponiendo que el directorio raíz está un nivel arriba del directorio 'data_exploration'
project_root = os.path.join(script_dir, '..')  # 'os.path.join' construye una ruta subiendo un nivel
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



# # Example usage
# if __name__ == "__main__":
#     start_date = "2023-12-01 00:00"
#     end_date = "2023-12-14 00:00"
#     stations = ['UAX', 'MER', 'XAL']
#     pollutant = 'cont_otres'

#     data = pollutant_by_stations(start_date, end_date, stations, pollutant)
#     print(data)
# %%

import pandas as pd

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

# # Example usage
# if __name__ == "__main__":
#     start_date = "2023-12-01 00:00"
#     end_date = "2023-12-14 00:00"
#     stations = ['UAX', 'MER', 'XAL']
#     pollutant = 'cont_otres'

#     data = daily_max_pollutant_by_stations(start_date, end_date, stations, pollutant)
#     print(data)



# %%
# import pandas as pd

# def date_above_threshold(start_date: str, end_date: str, stations: list, pollutant: str, threshold: float) -> pd.DataFrame:
#     """
#     Get entries where pollution data exceed a specified threshold for specific stations and pollutant within a date range.

#     Parameters:
#         start_date (str): The start date in the format "YYYY-MM-DD HH:MM".
#         end_date (str): The end date in the format "YYYY-MM-DD HH:MM".
#         stations (list): A list of station codes.
#         pollutant (str): The specific pollutant to query, e.g., "cont_otres" or "cont_pmdoscinco".
#         threshold (float): The threshold value above which to identify entries.

#     Returns:
#         pd.DataFrame: A DataFrame where each row represents an entry that exceeded the threshold,
#                        including the timestamp, the station code, and the value.
#     """
#     # First, get the complete data set using the previously defined function
#     pollution_data = pollutant_by_stations(start_date, end_date, stations, pollutant)

#     # Prepare a DataFrame to collect the results
#     results = []

#     # Filter entries that exceed the threshold for each station
#     for station in stations:
#         if station in pollution_data.columns:  # Check if the station has data
#             # Filter data where pollution exceeds the threshold
#             filtered_data = pollution_data[pollution_data[station] > threshold]
#             for timestamp, value in filtered_data[station].items():
#                 results.append({'date': timestamp, 'station': station, 'value': value})

#     # Convert list of dictionaries to DataFrame
#     results_df = pd.DataFrame(results)
#     return results_df

# # Example usage

# start_date = "2022-12-01 00:00"
# end_date = "2023-12-14 00:00"
# stations = ['UAX', 'MER', 'XAL']
# pollutant = 'cont_otres'
# threshold = 90

# data = date_above_threshold(start_date, end_date, stations, pollutant, threshold)
# print(data)

# %% Alternativa a  función date_above_threshold...


# Supongamos que la función pollutant_by_stations ya ha sido definida y está disponible.

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

# Ejemplo de uso
start_date = "2010-12-01 00:00"
end_date = "2023-12-14 00:00"
stations = ['UAX', 'MER', 'XAL']
pollutant = 'cont_otres'

df = pollutant_by_stations(start_date, end_date, stations, pollutant)
df_above_th = dates_above_threshold(df, 95.0)

print(df_above_th)


# %%

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def average_meteo(start_date, hours, field):
    """
    Calculate the hourly average for a specified meteorological field from a starting date over a specified number of hours.

    Parameters:
        start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
        hours (int): The number of hours to average over from the start date.
        field (str): The meteorological field to average (e.g., 'temp', 'u', 'v').

    Returns:
        pd.DataFrame: A DataFrame with the average values of the specified field over the given number of hours.
    """
    # Path to the WRF data file, adjust as necessary
    wrf_file_path = "/ServerData/WRF_2017_Kraken/wrfout_d01_2017-01-01_00.nc"  # Example file path

    # Load the data using xarray
    ds = xr.open_dataset(wrf_file_path)

    # Convert start date string to datetime object
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M")

    # Calculate the end datetime
    end_datetime = start_datetime + timedelta(hours=hours)

    # Select the data for the given field and time range
    data = ds[field].sel(time=slice(start_datetime, end_datetime))

    # Calculate the hourly average
    hourly_data = data.resample(time='1H').mean(dim='time')

    # Convert to DataFrame
    hourly_df = hourly_data.to_dataframe().reset_index()

    # Drop unnecessary columns if exist and just leave the 'time' and field value columns
    hourly_df = hourly_df[['time', field]]

    return hourly_df

# Example usage
start_date = "2017-01-01 00:00"
hours = 24
field = 'T2'  # Replace with your actual field name, e.g., temperature, wind u-component
average_values = average_meteo(start_date, hours, field)
print(average_values)

# %%



# %%
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from proj_io.inout import get_month_folder_esp
from os.path import join
from proj_preproc.wrf import crop_variables_xr, subsampleData

import matplotlib.pyplot as plt  # Para crear figuras y subfiguras
import cartopy.crs as ccrs       # Para manejar las proyecciones cartográficas
import numpy as np               # Para manejar arrays, usado aquí implícitamente para manipular datos


wrf_input_folder = "/ServerData/WRF_2017_Kraken"
bbox =  [18.75, 20,-99.75, -98.5]
grid_size_wrf = 4  # 4 for 4x4

def average_meteo(start_date, hours, field):
    """
    Calculate the hourly average for a specified meteorological field from a starting date over a specified number of hours,
    correcting for GMT.

    DOING: Ya obtiene el archivo del día, el campo para las diferentes horas, y el indice de la hora de inicio
    requerida de ese día. 

    TODO: USAR ÍNDICE del Hora inicial, y sacar el valor promedio para las siguientes (i.e. 3) horas para un valor promediado del grid.



    Parameters:
        start_date (str): The starting date and time in the format "YYYY-MM-DD HH:MM".
        hours (int): The number of hours to average over from the start date.
        field (str): The meteorological field to average (e.g., 'T2', 'U10', 'V10').

    Returns:
        pd.DataFrame: A DataFrame with the average values of the specified field over the given number of hours.
    """
    # Parse the starting date
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    gmt_correction = -6  # Correcting to match UTC time used by WRF data
    start_datetime_wrf = start_datetime + timedelta(hours=gmt_correction)
    year = start_datetime_wrf.year
    month = start_datetime_wrf.month
    
    cur_month = get_month_folder_esp(month) # get folder in spanhish names spec
    wrf_file_name = f"wrfout_d01_{start_datetime.strftime('%Y-%m-%d')}_00.nc"
    wrf_file = join(join(wrf_input_folder,str(start_datetime.year), cur_month), wrf_file_name)

    days_before_wrf = 0 # Var to check several nc's, in case the last is not found
    wrf_times_to_load = range(72)  # 
    meteo_var_names = [field]

    while not(os.path.exists(wrf_file)):
        print(f"ERROR File {wrf_file} does not exist")
        days_before_wrf += 1
        wrf_run_day = start_datetime - timedelta(days=days_before_wrf)
        wrf_file_name = f"wrfout_d01_{wrf_run_day.strftime('%Y-%m-%d')}_00.nc"
        wrf_file = join(join(wrf_input_folder,str(start_datetime.year), cur_month), wrf_file_name)
        wrf_times_to_load = np.array(list(wrf_times_to_load)) + 24 # Modify reading 24 'shifted' hours. 
    print(f"Working with wrf file: {wrf_file}")
    
    cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
    cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=wrf_times_to_load)
    
    #Subsampling the data
    subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
    wrf_time_format = '%Y-%m-%d_%H:%M:%S'
    wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values[wrf_times_to_load]])
    first_time_idx = np.where(wrf_dates >= start_datetime_wrf)[0][0]
    print(f"Current time from wrf is {wrf_dates[first_time_idx]} (Original start time: {start_datetime} )")
    meteo_cols = {}

    for cur_var_name in meteo_var_names:
            for cur_hour, wrf_hour_index in enumerate(range(first_time_idx, first_time_idx+hours)):
                cur_var_np = subsampled_xr_ds[cur_var_name].values
                var_flat_values = np.array(cur_var_np[wrf_hour_index,:,:].flatten())
                temp = {f'{cur_var_name}_{i}_h{cur_hour}':val for i, val in enumerate(var_flat_values)} 
                meteo_cols = {**meteo_cols, **temp} # Se arma diccionario con campos grid, y horas 
            average_value = sum(meteo_cols.values()) / len(meteo_cols)
            print(f'average_value: {average_value}')
    

    # Calculate the end datetime considering GMT correction
    end_datetime = start_datetime_wrf + timedelta(hours=hours - 1)  # Inclusive of the start hour
    
    return average_value#hourly_df[['time', field]]

# Example usage

start_date = "2024-04-09 00:00"
hours = 1
field = 'T2'  # Example meteorological field, e.g., temperature
average_values = average_meteo(start_date, hours, field)
print(average_values)
# %%

import matplotlib.pyplot as plt
from datetime import datetime, timedelta


start_date = datetime.strptime("2024-04-09 00:00", "%Y-%m-%d %H:%M")
field = 'T2'  # Meteorological field
hours = 1     # Number of hours to average over
times = []
averages = []

# Calculate end date (5 days later)
end_date = start_date + timedelta(days=5)

# Current date starts at the start and proceeds in 3-hour intervals
current_date = start_date

while current_date < end_date:
    formatted_date = current_date.strftime("%Y-%m-%d %H:%M")
    average_value = average_meteo(formatted_date, hours, field)
    if average_value is not None:
        times.append(current_date)
        averages.append(average_value)
    current_date += timedelta(hours=3)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(times, averages, marker='o', linestyle='-', color='b')
plt.title('Temperature Averages Over Time')
plt.xlabel('Time')
plt.ylabel('Average Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# %%
