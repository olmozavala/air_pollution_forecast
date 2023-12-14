# %% 
import sys
import psycopg2
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocDBParams, getPreprocWRFParams
from db.queries import getPollutantFromDateRange, getAllStations
from db.sqlCont import getPostgresConn
from proj_preproc.wrf import crop_variables_xr, subsampleData
import xarray as xr
from datetime import date, datetime, timedelta
from sklearn import preprocessing
from conf.localConstants import constants
from pandas import DataFrame
import pandas as pd
import time
from ai_common.constants.AI_params import *
from os.path import join
from conf.params import LocalTrainingParams, PreprocParams
from conf.TrainingUserConfiguration import get_makeprediction_config
from AI.data_generation.utilsDataFormat import *
from ai_common.models.modelSelector import select_1d_model
from os import listdir
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from copy import deepcopy
from proj_io.inout import filter_data, add_previous_hours, add_forecasted_hours, generateDateColumns, get_column_names, read_wrf_files_names, get_month_folder_esp, save_columns
from proj_preproc.preproc import loadScaler
from proj_prediction.prediction import compile_scaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home
sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')
import warnings
# Suppress the specific PerformanceWarning from pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Cargar el CSV con los nombres de las columnas
csv_xcols_path = '/ZION/AirPollutionData/Data/TrainingTestsPS/4paper_test01_15x3_5btsp_24ph_v20_2010_2019/X_columns.csv'
column_names_df = pd.read_csv(csv_xcols_path, header=None)
column_names = column_names_df.iloc[:, 0].tolist()
column_names.remove('0')

#forecast_time = datetime(2023, 1, 12, 14, 0, 0, 0)
#forecast_time = datetime.now().replace(minute=0, second=0, microsecond=0)
# %%  Read time from command line
forecasted_times = pd.date_range(start='2023-03-25 06:00:00', end='2025-12-01', freq='H')
forecast_time = forecasted_times[0]
print(f"################################# We will try to forecast time: {forecast_time.strftime('%Y-%m-%d %H:%M:%S')} #######################")

# %% 
config = get_makeprediction_config()
# *********** Reads the parameters ***********
model_weights_file = '/ZION/AirPollutionData/Data/TrainingTestsPS/test01_15x3_5btsp_24ph_v20_2010_2019/models/TestsPS_4paper_otres_2023_10_26_20_45-epoch-47-loss-0.25776392.hdf5'
file_name_norm = file_name_norm = "/ZION/AirPollutionData/Data/TrainingTestsPS/test01_15x3_5btsp_24ph_v20_2010_2019/norm/TestsPS_4paper_otres_2023_10_26_20_39_scaler.pkl"
cur_pollutant = "otres"
hours_before = 24  # Hours previous from pollution info
forecasted_hours = 24  # Forecasted hours from 1 to 24
grid_size_wrf = 4  # 4 for 4x4

wrf_config = getPreprocWRFParams()
meteo_var_names = wrf_config[PreprocParams.variables]
wrf_input_folder = "/ServerData/WRF_2017_Kraken"
bbox = wrf_config[PreprocParams.bbox]
times = wrf_config[PreprocParams.times]
# print(bbox)

all_stations = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]

# %% Creat an empty dataframe with proper column names 
contaminant_columns = [f'cont_{cur_pollutant}_{station}' for station in all_stations]
prev_contaminant_columns = [f'minus_{hour+1:02d}_{x}' for hour in range(hours_before)  for x in contaminant_columns]
all_meteo_columns = [f'{meteo_var}_{loc}_h{hour}' for hour in range(0, forecasted_hours) for meteo_var in meteo_var_names for loc in range(16) ]
all_time_colums = ['half_sin_day','half_cos_day','half_sin_year','half_cos_year','half_sin_week','half_cos_week',
                'sin_day','cos_day','sin_year','cos_year','sin_week','cos_week']

y_cols = [f'plus_{hour+1:02d}_{x}' for hour in range(forecasted_hours)  for x in contaminant_columns]

print(f"Total number of columns: {len(contaminant_columns)}(cont) + {len(prev_contaminant_columns)}(prev cont) {len(all_meteo_columns)}(meteo) {len(all_time_colums)}(time)")
print(f"Total number of columns: {len(contaminant_columns) + len(prev_contaminant_columns) + len(all_meteo_columns) + len(all_time_colums)}")
print(f"stations {len(contaminant_columns)} prevhours {len(prev_contaminant_columns)} meteo: {len(all_meteo_columns)} time: {len(all_time_colums)}")
print(f"Output columns total: {len(y_cols)}")
date_format = '%Y-%m-%d:%H'
timestamp_format = '%Y-%m-%d %H:%M:%S'

# %% --------------------- Read from database (current values and previous ones) 
end_time_df = forecast_time + timedelta(hours=hours_before+3)  # We are adding these 3 just for safety
print(F"Getting data from {forecast_time.strftime(date_format)} to {end_time_df.strftime(date_format)}")

conn = getPostgresConn()
res_cur_data = getPollutantFromDateRange(conn, f'cont_{cur_pollutant}', forecast_time, end_time_df, all_stations)
cur_data = np.array(res_cur_data)
dates = np.unique(np.array([x[0] for x in cur_data]))
# conn.close()
# We update the end datetime to the last one we got from the database
forecast_time = dates[0]
print(f"We will try to forecast time: {forecast_time.strftime(timestamp_format)}")
start_datetime = forecast_time 

# Define an empty X input to our model (we could validate all the sizes are correct here)
X = DataFrame(index=[forecast_time], columns=contaminant_columns+all_time_colums+all_meteo_columns+prev_contaminant_columns)

# Create empty dataframe with proper dimensions
if len(dates) < hours_before:
    raise Exception(f"Not enough dates were found len(dates)={len(dates)}")

print("Done!")
# %% 
# Template dataframe
print("Making empty dataset...")
df = DataFrame(index=dates[0:hours_before], columns=[f'cont_{cur_pollutant}_{station}' for station in all_stations])

for c_date in dates: # We are adding 1 because we want to get the current hour
    for station_idx, c_station in enumerate(all_stations):
        c_date_data = cur_data[cur_data[:,0] == c_date] # Data from the database for current date
        desired_data = c_date_data[c_date_data[:,2] == c_station] # If the station is the same as the one in the database
        if desired_data.shape[0] > 0:
            df.loc[c_date.strftime(timestamp_format),f'cont_{cur_pollutant}_{c_station}'] = desired_data[0,1]
# print(df.head())
print("Done!")

# %% ----------------------- Read from WRF forecast (current values and next ones) 
print("Adding meteorological data...")
gmt_correction = -6 # con -6 se obtiene start_datetime_wrf  más 6 horas que sería la hora en UTC. porlo que parece -6 ciertamente corrige.
start_datetime_wrf = forecast_time + timedelta(hours=gmt_correction)  # We are adding these 3 just for safety 

cur_month = get_month_folder_esp(start_datetime_wrf.month)

# Si leimos el mismo dia
wrf_idx = forecast_time.hour - gmt_correction
# Si no leimos el mismo dia entonces debe ser 
# wrf_idx = 24 - gmt_correction + forecast_time.hour

# Move outside here
wrf_file_name = f"wrfout_d01_{forecast_time.strftime('%Y-%m-%d')}_00.nc"
wrf_file = join(join(wrf_input_folder,str(forecast_time.year), cur_month), wrf_file_name)
print(f"Working with wrf file: {wrf_file}")
if not(os.path.exists(wrf_file)):
    raise Exception(f"File {wrf_file} does not exist")

cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=times)

# %% Subsampling the data
subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)

# %% type by columns orignal the all Getting ====== 
all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X)
print(f"Number of columns total: {X.shape[1]} contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")

# %% pollutants other for columns Remove ====== 
print("Filtering single pollutant..")
X_filt = filter_data(X, filter_type='single_pollutant', filtered_pollutant=cur_pollutant) 
all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X_filt)
print(f"(After) Number of not nan values in input data {X_filt.count().sum()}/{X_filt.shape[1]}")
print(f"(After) contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")

# %% -------- Making plots ---------
station = "UIZ"
# Filter dataframe with columns containing MER
df_mer = df.filter(regex=f'cont_{cur_pollutant}_{station}')

fig, axs = plt.subplots(5, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
# Plot mean values of temperature
next_hours = 24
axs[0,0].plot(df_mer[:next_hours], label='Ozono', c='r')
axs[0,0].legend()
# Transform kelvin to celcius
axs[1,0].plot(df_mer[:next_hours].index, subsampled_xr_ds['T2'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)) - 273.15, label='T2')
axs[1,0].legend()

axs[2,0].plot(df_mer[:next_hours].index, subsampled_xr_ds['V10'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='V10')
axs[2,0].legend()
axs[2,1].plot(df_mer[:next_hours].index, subsampled_xr_ds['U10'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='U10')
axs[2,1].legend()

axs[3,0].plot(df_mer[:next_hours].index, subsampled_xr_ds['RAINC'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='RAINC')
axs[3,0].legend()
axs[3,1].plot(df_mer[:next_hours].index, subsampled_xr_ds['RAINNC'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='RAINNC')
axs[3,1].legend()

axs[4,0].plot(df_mer[:next_hours].index, subsampled_xr_ds['SWDOWN'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='SWDOWN')
axs[4,0].legend()
axs[4,1].plot(df_mer[:next_hours].index, subsampled_xr_ds['GLW'].values[wrf_idx:wrf_idx + next_hours,:,:].mean(axis=(1,2)), label='GLW')
axs[4,1].legend()

plt.show()
# %%
