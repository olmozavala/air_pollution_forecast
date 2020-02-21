import numpy as np
from datetime import datetime, date, timedelta
import calendar
import pandas as pd
from inout.io_common import  create_folder
from os.path import join

from conf.localConstants import constants

# Manually setting the min/max values for the pollutant (ozone)
_min_value_ozone = 0
_max_value_ozone = 250

def generate_date_hot_vector(datetimes_original):
    # datetimes = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in datetimes_original]
    datetimes = datetimes_original
    day_hv = 7
    hour_hv = 24
    year_v = 1  # In this case we will generate a value from 0 to 1 where 0 is 1980 and 1 is 2050
    min_year = 1980
    max_year = 2040

    out_dates_hv = np.zeros((len(datetimes), day_hv+hour_hv+year_v))
    for i, cur_dt in enumerate(datetimes):
        year_v = (cur_dt.year - min_year)/(max_year-min_year)
        week_day = calendar.weekday(cur_dt.year, cur_dt.month, cur_dt.day)
        day_hv = [1 if x == week_day else 0 for x in range(7)]
        hour = cur_dt.hour
        hour_hv = [1 if x == hour else 0 for x in range(24)]
        out_dates_hv[i,0] = year_v
        out_dates_hv[i,1:8] = day_hv
        out_dates_hv[i,8:35] = hour_hv

    day_strs = [F'week_{x}' for x in range(7)]
    hour_strs = [F'hour_{x}' for x in range(24)]
    column_names = ['year'] + day_strs + hour_strs
    dates_hv_df = pd.DataFrame(out_dates_hv, columns=column_names, index=datetimes_original)
    return dates_hv_df

def normalizeAndFilterData(data, datetimes_orig, forecasted_hours, output_folder='', run_name='', read_from_file=False):
    """
    This function normalizes de data and filters only the cases where we
    have the appropiate forecasted times. It also obtains the 'y' index
    :param data: All the data
    :param datetimes_str: An array of datetimes as strings which correspond to the index
    :param forecasted_hours: an integer representing the number of hours in advance we want to read
    :return:
    """
    # Predicting for the next value after 24hrs (only one)
    print("Normalizing data....")
    datetimes = np.array(datetimes_orig)

    all_data_cols = data.columns.values
    date_columns = [x for x in all_data_cols if (x.find('week') != -1) or (x.find('hour') != -1) or (x.find('year') != -1)]
    stations_columns = [x for x in all_data_cols if (x.find('h') == -1) and (x not in date_columns)]
    meteo_columns = [x for x in all_data_cols if (x.find('h') != -1) and (x not in date_columns)  and (x not in stations_columns)]

    # Normalizing meteorological variables
    # In this case we obtain the normalization values directly from the data
    if not(read_from_file):
        min_values_meteo = data[meteo_columns].min()
        max_values_meteo = data[meteo_columns].max()
        # ********* Saving normalization values for each variable ******
        create_folder(output_folder)
        min_values_meteo.to_csv(join(output_folder,F'{run_name}_min_values.csv'))
        max_values_meteo.to_csv(join(output_folder,F'{run_name}_max_values.csv'))
    else: # In this case we obtain the normalization values from the provided file
        min_values_meteo = pd.read_csv(join(output_folder,F'{run_name}_min_values.csv'), names=['Min'], squeeze=True)
        max_values_meteo = pd.read_csv(join(output_folder,F'{run_name}_max_values.csv'), names=['Max'], squeeze=True)

    data_norm_df = data.copy()
    data_norm_df[meteo_columns] = (data_norm_df[meteo_columns] - min_values_meteo)/(max_values_meteo - min_values_meteo)
    data_norm_df[stations_columns] = (data_norm_df[stations_columns] - _min_value_ozone)/(_max_value_ozone - _min_value_ozone)
    print(F'Done!')

    # Filtering only dates where there is data "forecasted hours after" (24 hrs after)
    print(F"Building X and Y ....")
    accepted_times_idx = []
    y_times_idx = []
    for i, c_datetime in enumerate(datetimes):
        forecasted_datetime = c_datetime + np.timedelta64(forecasted_hours,'h')
        if forecasted_datetime in datetimes:
            accepted_times_idx.append(i)
            y_times_idx.append(np.argwhere(forecasted_datetime == datetimes)[0][0])

    # ****************** Replacing nan columns with the mean value of all the other columns ****************
    mean_values = data_norm_df[stations_columns].mean(1)

    # TODO aqui mero hay que poner -1 donde no haya datos y generar otra columna que se llame mean
    # El reemplazo se tendra que hacer cuando se generan las X y las Y

    print(F"Filling nan values....")
    data_norm_df_final = data_norm_df.copy()
    for cur_station in stations_columns:
        data_norm_df_final[cur_station] = data_norm_df[cur_station].fillna(mean_values)

    # print(F"Norm params: {scaler.get_params()}")
    # file_name_normparams = join(parameters_folder, F'{model_name}.txt')
    # utilsNN.save_norm_params(file_name_normparams, NormParams.min_max, scaler)
    print("Done!")

    return data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns

def deNormalize(data):
    unnormalize_data = data*(_max_value_ozone- _min_value_ozone) + _min_value_ozone
    return unnormalize_data

