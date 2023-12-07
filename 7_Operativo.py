# %% trusted=true
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
matplotlib.use('Agg')
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
from viz.figure_generator import plot_input_output_data
from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import  create_folder
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

def operativo(forecast_time):
    # %% trusted=true
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

    # %% Creat an empty dataframe with proper column names trusted=true
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
    print(f"################################# We will try to forecast time: {forecast_time.strftime(timestamp_format)} #######################")

    try:
        # %% --------------------- Read from database (current values and previous ones) trusted=true
        start_datetime_df = forecast_time - timedelta(hours=hours_before+3)  # We are adding these 3 just for safety
        print(F"Getting data from {start_datetime_df.strftime(date_format)} to {forecast_time.strftime(date_format)}")

        conn = getPostgresConn()
        res_cur_data = getPollutantFromDateRange(conn, f'cont_{cur_pollutant}', start_datetime_df, forecast_time, all_stations)
        cur_data = np.array(res_cur_data)
        dates = np.unique(np.array([x[0] for x in cur_data]))
        # print(cur_data)
        # print(dates)
        # conn.close()
        # We update the end datetime to the last one we got from the database
        forecast_time = dates[-1]
        print(f"We will try to forecast time: {forecast_time.strftime(timestamp_format)}")
        start_datetime = forecast_time - timedelta(hours=hours_before)

        # Define an empty X input to our model (we could validate all the sizes are correct here)
        X = DataFrame(index=[forecast_time], columns=contaminant_columns+all_time_colums+all_meteo_columns+prev_contaminant_columns)
        # Saving the columsn should be in order
        # save_columns(X, 'X_columns.csv')

        # Create empty dataframe with proper dimensions
        if len(dates) < hours_before:
            #  
            # if len(dates) <= 20: #considerar para en caso de algo más flexible... faltando 4 horas en db.
            raise Exception(f"Not enough dates were found len(dates)={len(dates)}")

        print("Done!")
        # %% trusted=true
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

        # %% trusted=true
        print("Adding previous hours...")
        df_shifted = add_previous_hours(df, hours_before=hours_before)
        # print(df_shifted.head())
        print("Done!")

        # %% trusted=true
        print("Updating our template dataframe with pollution data...")
        print(f"(Before) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")
        X.update(df_shifted)
        print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")
        print("Done!")
        # %% Add time columns here trusted=true
        print("Adding time columns...")
        time_cols, time_values = generateDateColumns(datetimes=[forecast_time])
        X.update(DataFrame(index=[forecast_time], columns=time_cols, data=np.array([x[0] for x in time_values]).reshape(1,len(time_values))))
        print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")

        # %% ----------------------- Read from WRF forecast (current values and next ones) trusted=true
        print("Adding meteorological data...")
        gmt_correction = -6 # con -6 se obtiene start_datetime_wrf  más 6 horas que sería la hora en UTC. porlo que parece -6 ciertamente corrige.
        start_datetime_wrf = forecast_time - timedelta(hours=gmt_correction)  # We are adding these 3 just for safety 
        
        cur_month = get_month_folder_esp(start_datetime_wrf.month)
        
        # Move outside here
        wrf_file_name = f"wrfout_d01_{forecast_time.strftime('%Y-%m-%d')}_00.nc"
        wrf_file = join(join(wrf_input_folder,str(forecast_time.year), cur_month), wrf_file_name)
        print(f"Working with wrf file: {wrf_file}")
        if not(os.path.exists(wrf_file)):
            raise Exception(f"File {wrf_file} does not exist")


        cur_xr_ds = xr.open_dataset(wrf_file, decode_times=False)
        cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, meteo_var_names, bbox, times=times)
        # cropped_xr_ds= cur_xr_ds.sel(XLAT=slice(bbox[0], bbox[1]), XLONG=slice(bbox[2], bbox[3]))
        # newLAT  = cropped_xr_ds['XLAT'].values
        # newLon  = cropped_xr_ds['XLONG'].values

        # # %% Plot using caartopy show country limits
        # bbox = [newLon[0], newLon[-1], newLAT[0], newLAT[-1]]
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        # ax.set_extent(bbox)
        # ax.add_feature(cfeature.COASTLINE)
        # shape_feature = ShapelyFeature(Reader("data/MEX_adm1.shp").geometries(), ccrs.PlateCarree(), 
        # edgecolor='black', facecolor='none')
        # ax.add_feature(shape_feature)
        # im = ax.imshow(cropped_xr_ds['T2'].values[0,:,:], extent=bbox, origin='lower', transform=ccrs.PlateCarree(), cmap='coolwarm')
        # gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
        # ax.add_feature(shape_feature, edgecolor='black', facecolor='none')
        # gl.top_labels = False
        # gl.left_labels = False
        # gl.xlabel_style = {'size': 10, 'weight':'bold'}
        # # plt.colorbar(im, location='right', shrink=.6, pad=.12)
        # plt.show()

# %% trusted=true

        # %% Subsampling the data
        subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, meteo_var_names, grid_size_wrf, grid_size_wrf)
        wrf_time_format = '%Y-%m-%d_%H:%M:%S'
        wrf_dates = np.array([datetime.strptime(x.decode('utf-8'), wrf_time_format) for x in cur_xr_ds['Times'].values])
        # print(f"Original forecast time: {forecast_time} \n WRF dates in file: {wrf_dates}")
        first_time_idx = np.where(wrf_dates >= start_datetime_wrf)[0][0]
        print(f"Assuming current time from wrf is {wrf_dates[first_time_idx]} (Original forecast time: {forecast_time} )")
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


        temp_df = DataFrame(index=[forecast_time], data=meteo_cols)
        assert temp_df.index == X.index
        X.update(temp_df)
        # print(f"(After) Number of not nan values in input data {X.count().sum()}/{X.shape[1]}")

        # ------------------------------ FINALLY WE HAVE FULL INPUT X -----------------------------------
        
        # %% type by columns orignal the all Getting ====== trusted=true
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X)
        print(f"Number of columns total: {X.shape[1]} contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")

        # %% pollutants other for columns Remove ====== trusted=true
        print("Filtering single pollutant..")
        X_filt = filter_data(X, filter_type='single_pollutant', filtered_pollutant=cur_pollutant) 
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(X_filt)
        print(f"(After) Number of not nan values in input data {X_filt.count().sum()}/{X_filt.shape[1]}")
        print(f"(After) contaminant: {len(all_contaminant_columns)} Meteo: {len(all_meteo_columns)} Time:{len(all_time_colums)}")
        # %% ------ reordering X_filt        
        # Suponiendo que 'X' es tu DataFrame existente
        ################
        # Ordenar las columnas de 'X' para que coincidan con el orden en el CSV
        X_filt = X_filt[column_names]
        ################
        # %% -------- Normalizing data trusted=true
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
        # X_filt.to_csv(f"{forecast_time.strftime('%Y_%m_%d_%H')}.csv", index=False)
        data_norm_np = scaler.transform(X_filt)
        X_norm = DataFrame(data_norm_np, columns=X.columns, index=X.index)

        print(F'X {X_norm.shape}, Memory usage: {X_norm.memory_usage().sum()/1024**2:02f} MB')

        # %% trusted=true
        print("Removing time index...")
        X_norm.reset_index(drop=True, inplace=True)

        # %% Replace all the nan values with another value trusted=true
        replace_value = 0
        print(f"Replacing nan values with {replace_value}...")
        X_norm.fillna(replace_value, inplace=True)

        # %% ******************* Selecting the model ********************** trusted=true
        config[ModelParams.INPUT_SIZE] = X_norm.shape[1]
        config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = len(all_stations)*forecasted_hours

        print('Reading model ....')
        model = select_1d_model(config)

        # *********** Reads the weights***********
        print('Reading weights ....')
        model.load_weights(model_weights_file)
        print("Done!")

        # %% Calculo de predicciones de el dataset de test: trusted=true
        print('Making prediction ....')

        Y_pred = model.predict(X_norm.values)
        print("Done!")

# %% A function is defined to generate custom scaler objects trusted=true
        # %%
        print("Denormalizing...")
        scaler_y = generate_scaler(scaler,y_cols)
        print("Done ...")

        # %% Desescalar las predicciones trusted=true
        Y_pred_descaled = scaler_y.inverse_transform(Y_pred)

        # Convertir Y_pred_descaled en un DataFrame de pandas
        y_pred_descaled_df = pd.DataFrame(Y_pred_descaled, columns=scaler_y.feature_names_in_)
        # Verificar el DataFrame
        print(y_pred_descaled_df.head())

        # %% Plot results trusted=true
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
# %% trusted=true
        #%%
        all_hours = ','.join([f'hour_p{x:02d}' for x in range(1,25)])
        # print(all_hours)

        for c_station in all_stations:

            try: 
                conn = getPostgresConn()
                # TODO we need to verify if there is already a forecast for this time
                pred_by_station = y_pred_descaled_df.filter(regex=c_station)
                # print('Aquí ok 0')
                sql = f"""INSERT INTO forecast_otres (fecha, id_tipo_pronostico, id_est, val, {all_hours}) 
                            VALUES (%s, %s, %s, %s, {','.join(['%s']*24)})"""

                # print(sql)
                cur = conn.cursor()
                cur.execute(sql, (forecast_time, '6', c_station, '-1', *[f'{x:0.2f}' for x in pred_by_station.iloc[0].tolist()],))
                cur.close()
                conn.commit()
                conn.close()
            except Exception as e:
                print(f'Error inserting into DB: {e}')
            finally:
                conn.close()
    #%%
    except Exception as e:
        print(f"Error: Failed for {forecast_time.strftime(date_format)} with error {e}")

# %% --------- Running the operativo trusted=true
import time

def waiting_with_updates(total_minutes, update_interval):
    """funcion que pausa, e indica cuanto falta por esperar
    total_minutes and update_interval in minutes
    i.e. waiting_with_updates(60, 10)"""

    for remaining in range(total_minutes, 0, -update_interval):
        print(f'Waiting for {remaining} minutes...')
        time.sleep(update_interval * 60)


if __name__ == "__main__":
    # while True:
    #     forecast_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    #     print(forecast_time)
    #     operativo(forecast_time)
    #     waiting_with_updates(60, 10)
    
    forecasted_times = pd.date_range(start='2019-01-03 00:00:00', end='2019-01-05', freq='H')
    for c_forecast_time in forecasted_times:
        operativo(c_forecast_time)