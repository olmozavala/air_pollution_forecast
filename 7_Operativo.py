# %%
# Standard imports
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta

# 3rd-p imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

# Local imports
from ai_common.constants.AI_params import *
from ai_common.models.modelSelector import select_1d_model
from conf.MakeWRF_and_DB_CSV_OperativoConfiguration import getPreprocWRFParams
from conf.TrainingOperativoConfiguration import get_makeprediction_config
from conf.params import PreprocParams
from db.queries import getPollutantFromDateRange
from db.sqlCont import getPostgresConn
from os.path import join
from pandas import DataFrame
from proj_io.inout import filter_data, add_previous_hours, generateDateColumns, get_column_names, get_month_folder_esp
from proj_preproc.preproc import loadScaler
from proj_preproc.wrf import crop_variables_xr, subsampleData
from AI.data_generation.utilsDataFormat import *

import warnings

# Suppress the specific PerformanceWarning from pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths for model processing and predictions 

# csv_xcols_path = '/ZION/AirPollutionData/Data/TrainingTestsPS/test01_15x3_5btsp_24ph_v20_2010_2019/X_columns.csv'
# model_weights_file = '/ZION/AirPollutionData/Data/TrainingTestsPS/test01_15x3_5btsp_24ph_v20_2010_2019/models/TestsPS_4paper_otres_2023_10_26_20_45-epoch-47-loss-0.25776392.hdf5'
# file_name_norm = "/ZION/AirPollutionData/Data/TrainingTestsPS/test01_15x3_5btsp_24ph_v20_2010_2019/norm/TestsPS_4paper_otres_2023_10_26_20_39_scaler.pkl"

csv_xcols_path = './operativo_files/test01_15x3_5btsp_24ph_v20_2010_2019/X_columns.csv'
model_weights_file = './operativo_files/test01_15x3_5btsp_24ph_v20_2010_2019/models/TestsPS_4paper_otres_2023_10_26_20_45-epoch-47-loss-0.25776392.hdf5'
file_name_norm = "./operativo_files/test01_15x3_5btsp_24ph_v20_2010_2019/norm/TestsPS_4paper_otres_2023_10_26_20_39_scaler.pkl"

wrf_input_folder = "/ServerData/WRF_2017_Kraken"

# Fixing parms used for training
cur_pollutant = "otres"
all_stations = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                    "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                    "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]
                    
hours_before = 24  # Hours previous from pollution info
forecasted_hours = 24  # Forecasted hours from 1 to 24
grid_size_wrf = 4  # 4 for 4x4

# *********** Reads configfiles parameters ***********
config = get_makeprediction_config()
wrf_config = getPreprocWRFParams()
meteo_var_names = wrf_config[PreprocParams.variables]    
bbox = wrf_config[PreprocParams.bbox]
wrf_times_to_load = wrf_config[PreprocParams.times]

# Cargar el CSV con los nombres de las columnas 
column_names_df = pd.read_csv(csv_xcols_path, header=None)
column_names = column_names_df.iloc[:, 0].tolist()
column_names.remove('0')

# %% Operativo function
def operativo(forecast_time):
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

    # %%
    print(f"We will try to forecast time: {forecast_time.strftime(timestamp_format)}")
    # %%
    try:
        # %% --------------------- Read from database (current values and previous ones)
        start_datetime_df = forecast_time - timedelta(hours=hours_before+3)  # We are adding these 3 just for safety
        print(F"Getting data from {start_datetime_df.strftime(date_format)} to {forecast_time.strftime(date_format)}")

        conn = getPostgresConn()
        res_cur_data = getPollutantFromDateRange(conn, f'cont_{cur_pollutant}', start_datetime_df, forecast_time, all_stations)
        cur_data = np.array(res_cur_data)
        dates = np.unique(np.array([x[0] for x in cur_data]))
        print(cur_data)
        print(dates)
        conn.close()
        # We update the end datetime to the last one we got from the database
        forecast_time = dates[-1]
        print(f"We will try to forecast time: {forecast_time.strftime(timestamp_format)}")
        start_datetime = forecast_time - timedelta(hours=hours_before)

        # Define an empty X input to our model (we could validate all the sizes are correct here)
        X = DataFrame(index=[forecast_time], columns=contaminant_columns+all_time_colums+all_meteo_columns+prev_contaminant_columns)

        # Create empty dataframe with proper dimensions
        if len(dates) < hours_before:
            #  
            # if len(dates) <= 20: #considerar para en caso de algo más flexible... faltando 4 horas en db.
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

        # %%
        print("Adding prevoius hours...")
        df_shifted = add_previous_hours(df, hours_before=hours_before)
        # print(df_shifted.head())
        print("Done!")

        # %%
        print("Updating our template dataframe with pollution data...")
        print(f"(Before) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")
        X.update(df_shifted)
        print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")
        print("Done!")
        # %% Add time columns here
        print("Adding time columns...")
        time_cols, time_values = generateDateColumns(datetimes=[forecast_time])
        X.update(DataFrame(index=[forecast_time], columns=time_cols, data=np.array([x[0] for x in time_values]).reshape(1,len(time_values))))
        print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")

        # %% ----------------------- Read from WRF forecast (current values and next ones)
        print("Adding meteorological data...")
        gmt_correction = -6 # con -6 se obtiene start_datetime_wrf  más 6 horas que sería la hora en UTC. porlo que parece -6 ciertamente corrige.
        start_datetime_wrf = forecast_time - timedelta(hours=gmt_correction)  # We are adding these 3 just for safety 
        
        cur_month = get_month_folder_esp(start_datetime_wrf.month)
        
        # Move outside here
        wrf_file_name = f"wrfout_d01_{forecast_time.strftime('%Y-%m-%d')}_00.nc"
        wrf_file = join(join(wrf_input_folder,str(forecast_time.year), cur_month), wrf_file_name)
        days_before_wrf = 0
        # If the file does not exist, we will try to find the closest previous day with data from WRF
        wrf_times_to_load = wrf_config[PreprocParams.times]
        while not(os.path.exists(wrf_file)):
            print(f"ERROR File {wrf_file} does not exist")

            days_before_wrf += 1
            wrf_run_day = forecast_time - timedelta(days=days_before_wrf)
            wrf_file_name = f"wrfout_d01_{wrf_run_day.strftime('%Y-%m-%d')}_00.nc"
            wrf_file = join(join(wrf_input_folder,str(forecast_time.year), cur_month), wrf_file_name)
            wrf_times_to_load = np.array(list(wrf_times_to_load)) + 24 # Modify reading 24 'shifted' hours. 
        print(f"Working with wrf file: {wrf_file}")

        cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
        cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=wrf_times_to_load)

# %%
        # %% Subsampling the data
        subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
        wrf_time_format = '%Y-%m-%d_%H:%M:%S'
        wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values[wrf_times_to_load]]) # Here we generate new times from the original nc file dates
        # print(f"Original forecast time: {forecast_time} \n WRF dates in file: {wrf_dates}")
        first_time_idx = np.where(wrf_dates >= start_datetime_wrf)[0][0]
        print(f"Current time from wrf is {wrf_dates[first_time_idx]} (Original forecast time: {forecast_time} )")
        meteo_cols = {}
        for cur_var_name in meteo_var_names:
            for cur_hour, wrf_hour_index in enumerate(range(first_time_idx, first_time_idx+forecasted_hours)):
                cur_var_np = subsampled_xr_ds[cur_var_name].values
                var_flat_values = np.array(cur_var_np[wrf_hour_index,:,:].flatten())
                temp = {f'{cur_var_name}_{i}_h{cur_hour}':val for i, val in enumerate(var_flat_values)} 
                meteo_cols = {**meteo_cols, **temp}

                if cur_var_name == "T2" and cur_hour == 0:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                    im = ax.imshow(cur_var_np[0,:,:], extent=bbox, origin='lower', transform=ccrs.PlateCarree(), cmap='coolwarm')
                    print(cur_var_np.shape)
                    print(type(cur_var_np))
                    gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
                    #ax.add_feature(shape_feature, edgecolor='black', facecolor='none')
                    gl.top_labels = False
                    gl.left_labels = False
                    gl.xlabel_style = {'size': 10, 'weight':'bold'}
                    # plt.colorbar(im, location='right', shrink=.6, pad=.12)
                    plt.show()
                    plt.savefig("temp.jpg")


        temp_df = DataFrame(index=[forecast_time], data=meteo_cols)
        assert temp_df.index == X.index
        X.update(temp_df)
        # print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")

        # ------------------------------ FINALLY WE HAVE FULL INPUT X -----------------------------------

        # %% type by columns orignal the all Getting ======
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X)
        print(f"Number of columns total: {X.shape[1]} contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")

        # %% pollutants other for columns Remove ======
        print("Filtering single pollutant..")
        X_filt = filter_data(X, filter_type='single_pollutant', filtered_pollutant=cur_pollutant) 
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X_filt)
        print(f"(After) Number of not nan values in input data {X_filt.count().sum()}/{X_filt.shape[1]}")
        print(f"(After) contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")

        # %% ------ reordering X_filt
        ################
        # Ordenar las columnas de 'X' para que coincidan con el orden en el CSV

        X_filt = X_filt[column_names]

        ################
        # %% -------- Normalizing data
        # loading of original scaler object
        scaler_orig = loadScaler(file_name_norm)

        def generate_scaler(old_scaler, new_columns):
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
                if column.find('minus') != -1 or column.find('plus') != -1: # We will add these new colums as the 'original' scaler
                    original_column = '_'.join(column.split('_')[2:])  # We 'remove' the last 'minus_hr' part
                else:
                    original_column = column
                
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

        scaler = generate_scaler(scaler_orig, X_filt.columns)

        print("Normalizing data....")
        data_norm_np = scaler.transform(X_filt)
        X_norm = DataFrame(data_norm_np, columns=X.columns, index=X.index)

        print(F'X {X_norm.shape}, Memory usage: {X_norm.memory_usage().sum()/1024**2:02f} MB')

        # %%
        print("Removing time index...")
        X_norm.reset_index(drop=True, inplace=True)

        # %% Replace all the nan values with another value
        replace_value = 0
        print(f"Replacing nan values with {replace_value}...")
        X_norm.fillna(replace_value, inplace=True)

        # %% ******************* Selecting the model **********************
        config[ModelParams.INPUT_SIZE] = X_norm.shape[1]
        config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = len(all_stations)*forecasted_hours

        print('Reading model ....')
        model = select_1d_model(config)

        # *********** Reads the weights***********
        print('Reading weights ....')
        model.load_weights(model_weights_file)
        print("Done!")

        # %% Calculo de predicciones de el dataset de test:
        print('Making prediction ....')
        Y_pred = model.predict(X_norm.values)
        print("Done!")

# %% A function is defined to generate custom scaler objects
        # %%
        print("Denormalizing...")
        scaler_y = generate_scaler(scaler,y_cols)
        print("Done ...")

        # %% Desescalar las predicciones
        Y_pred_descaled = scaler_y.inverse_transform(Y_pred)

        # Convertir Y_pred_descaled en un DataFrame de pandas
        y_pred_descaled_df = pd.DataFrame(Y_pred_descaled, columns=scaler_y.feature_names_in_)
        # Verificar el DataFrame
        print(y_pred_descaled_df.head())

        # %% Plot results
        t = 30
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        for idx, c_station in enumerate(all_stations[0:t]):
            axs.plot(y_pred_descaled_df.filter(like=c_station).loc[0].values, label=c_station)
            axs.set_title(c_station)
            # Set the x axis as times starting from forecast_time
            axs.set_xticks(range(forecasted_hours))
            # axs.set_xticklabels([x.strftime(date_format) for x in [forecast_time + timedelta(hours=y) for y in range(forecasted_hours)]])
            axs.set_xticklabels([x.strftime("%H") for x in [forecast_time + timedelta(hours=y) for y in range(forecasted_hours)]])


        plt.legend()
        plt.title(f"Forecast for all stations at time {forecast_time.strftime(date_format)}")
        plt.show()
# %%
        # %%
        all_hours = ','.join([f'hour_p{x:02d}' for x in range(1,25)])
        # print(all_hours)

        for c_station in all_stations:

            try: 
                conn = getPostgresConn()
                pred_by_station = y_pred_descaled_df.filter(regex=c_station)
                print('Aquí ok 0')
                sql = f"""INSERT INTO forecast_otres (fecha, id_tipo_pronostico, id_est, val, {all_hours}) 
                            VALUES (%s, %s, %s, %s, {','.join(['%s']*24)})"""

                # print(sql)
                cur = conn.cursor()
                cur.execute(sql, (forecast_time, '6', c_station, '-1', *[f'{x:0.2f}' for x in pred_by_station.iloc[0].tolist()],))
                cur.close()
                conn.commit()
                conn.close()
            except Exception as e:
                print('Aquí no ok 0')
                print(e)
            finally:
                conn.close()
            print("Done")
    # %%
    except Exception as e:
        print(f"Error: Failed for {forecast_time.strftime(date_format)} with error {e}")

# %% --------- Running operativo

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Si se proporciona un argumento, usarlo
        try:
            forecast_time = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M")
        except ValueError:
            print("Incorrect date format. Please use YYYY-MM-DD HH:MM")
            exit(1)
    else:
        # Usar la hora actual si no se proporciona un argumento
        forecast_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    print(forecast_time)
    try:
        operativo(forecast_time)
        exit(0)
    except Exception as e:
        print(f"Error during operation: {e}")
        exit(1)


# Ejemplo de llamadas desde terminal:
# python 7_Operativo_refactor.py # > # date "+%Y-%m-%d %H:%M" | python 7_Operativo_refactor.py
# python 7_Operativo_refactor.py "2023-12-14 15:00"
