# %%

import sys
# sys.path.append('./eoas_pyutils')  # Doesn't work when using a conda env outside home
sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')
import os
import numpy as np
import pickle
from datetime import datetime, date, timedelta
import calendar
import pandas as pd
from ai_common.constants.AI_params import NormParams
from sklearn import preprocessing
from os.path import join
from pandas import DataFrame

from conf.localConstants import constants

# Manually setting the min/max values for the pollutant (ozone)
_min_value_ozone = 0
_max_value_ozone = 250


# %%
def generate_date_hot_vector(datetimes_original):
    # datetimes = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in datetimes_original]
    datetimes = datetimes_original
    week_day_hv = 7
    week_year_hv = 51
    hour_hv = 24
    year_hv = 1  # In this case we will generate a value from 0 to 1 where 0 is 1980 and 1 is 2050
    min_year = 1980
    max_year = 2040

    out_dates_hv = np.zeros((len(datetimes), week_day_hv + hour_hv + year_hv + week_year_hv))
    for i, cur_dt in enumerate(datetimes):
        year_hv = (cur_dt.year - min_year)/(max_year-min_year)
        week_day = calendar.weekday(cur_dt.year, cur_dt.month, cur_dt.day)
        week_day_hv = [1 if x == week_day else 0 for x in range(7)]
        hour = cur_dt.hour
        hour_hv = [1 if x == hour else 0 for x in range(24)]
        week_year = cur_dt.date().isocalendar()[1]
        week_year_hv = [1 if x == week_year else 0 for x in range(51)]
        out_dates_hv[i,0] = year_hv
        out_dates_hv[i,1:8] = week_day_hv
        out_dates_hv[i,8:32] = hour_hv
        out_dates_hv[i,32:83] = week_year_hv

    day_strs = [F'week_day_{x}' for x in range(7)]
    hour_strs = [F'hour_{x}' for x in range(24)]
    week_strs = [F'week_{x}' for x in range(51)]
    column_names = ['year'] + day_strs + hour_strs + week_strs
    dates_hv_df = pd.DataFrame(out_dates_hv, columns=column_names, index=datetimes_original)
    return dates_hv_df

# %%
def normalizeData(data, norm_type, file_name):
    if norm_type == NormParams.min_max:
        scaler = preprocessing.MinMaxScaler()
    if norm_type == NormParams.mean_zero:
        scaler = preprocessing.StandardScaler()

    scaler = scaler.fit(data)
    data_norm_np = scaler.transform(data)
    data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)

    # ******************* Saving Normalization params, scaler object **********************
    scaler.path_file = file_name
    with open(scaler.path_file, 'wb') as f: #scaler.path_file must be defined during training.
        pickle.dump(scaler, f)
    print(f'Scaler/normalizer object saved to: {scaler.path_file}')
    print(F'Done! Current shape: {data_norm_df.shape} ')
    return data_norm_df


def loadScaler(file_name):
    with open(file_name, 'rb') as f:
        scaler = pickle.load(f)
    
    return scaler

#Function to extract model_name from file names TODO: Pasar a proj_fun
def extract_model_name(models_path):
    # Get the list of files in the 'models' directory
    files = os.listdir(models_path)

    # Filter only the files with the model name
    model_files = [file for file in files if file.endswith('.hdf5')]

    if len(model_files) > 0:
        # Extract the model name from the first file (assuming all files have the same name prefix)
        model_name = model_files[0].split('-epoch')[0]
        return model_name
    else:
        return None


def deNormalize(data):
    unnormalize_data = data*(_max_value_ozone- _min_value_ozone) + _min_value_ozone
    return unnormalize_data


def apply_bootstrap(X_df, Y_df, contaminant, station, boostrap_threshold, forecasted_hours, boostrap_factor=1):
    '''
    This function will boostrap the data based on the threshold and the forecasted hours
    '''

    bootstrap_column = f"cont_{contaminant}_{station}"
    print("Bootstrapping the data...")
    # Searching all the index where X or Y is above the threshold

    # Adding index when the current time is above the threshold
    bootstrap_idx = X_df.loc[:,bootstrap_column] > boostrap_threshold

    # Searching index when any of the forecasted hours is above the threshold
    y_cols = Y_df.columns.values
    for i in range(1, forecasted_hours+1):
        # print(bootstrap_idx.sum())  
        c_column = f"plus_{i:02d}_{bootstrap_column}"
        if c_column in y_cols:
            bootstrap_idx = bootstrap_idx | (Y_df.loc[:, c_column] > boostrap_threshold)

    X_df = pd.concat([X_df, *[X_df[bootstrap_idx] for i in range(boostrap_factor)]])
    Y_df = pd.concat([Y_df, *[Y_df[bootstrap_idx] for i in range(boostrap_factor)]])

    return X_df, Y_df
