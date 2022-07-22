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
    # meteo_names = ['U10', 'V10', 'RAINC', 'T2', 'RAINNC', 'PBLH', 'SWDOWN', 'GLW']
    meteo_names = ['U10', 'V10', 'RAINC', 'T2', 'RAINNC', 'SWDOWN', 'GLW']
    if not(read_from_file):
        min_data = {}
        max_data = {}
        for cur_meteo in meteo_names:
            cur_meteo_cols = [x for x in meteo_columns if x.find(cur_meteo) != -1]
            min_data[cur_meteo] = data[cur_meteo_cols].min().min()
            max_data[cur_meteo] = data[cur_meteo_cols].max().max()
        # ********* Saving normalization values for each variable ******
        create_folder(output_folder)
        pd.DataFrame(min_data, index=[1]).to_csv(join(output_folder,F'{run_name}_min_values.csv'))
        pd.DataFrame(max_data, index=[1]).to_csv(join(output_folder,F'{run_name}_max_values.csv'))
    else: # In this case we obtain the normalization values from the provided file
        min_data = pd.read_csv(join(output_folder,F'{run_name}_min_values.csv'), index_col=0)
        max_data = pd.read_csv(join(output_folder,F'{run_name}_max_values.csv'), index_col=0)

    data_norm_df = data.copy()

    # Normalizing the meteorological variables
    for cur_meteo in meteo_names:
        cur_meteo_cols = [x for x in meteo_columns if x.find(cur_meteo) != -1]
        # The data structure is a little bit different when reading from the file
        if not (read_from_file):
            min_val = min_data[cur_meteo]
            max_val = max_data[cur_meteo]
        else:
            min_val = min_data[cur_meteo].values[0]
            max_val = max_data[cur_meteo].values[0]
        data_norm_df[cur_meteo_cols] = (data_norm_df[cur_meteo_cols] - min_val)/(max_val - min_val)

    # Normalizing the pollution variables
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

    # Replace nan values with -1 and add additional MEAN column
    print(F"Filling nan values....")
    data_norm_df_final = data_norm_df.copy()
    for cur_station in stations_columns:
        data_norm_df_final[cur_station] = data_norm_df[cur_station].fillna(-1)

    data_norm_df_final['MEAN'] = mean_values

    # print(F"Norm params: {scaler.get_params()}")
    # file_name_normparams = join(parameters_folder, F'{model_name}.txt')
    # utilsNN.save_norm_params(file_name_normparams, NormParams.min_max, scaler)
    print("Done!")

    return data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns

def deNormalize(data):
    unnormalize_data = data*(_max_value_ozone- _min_value_ozone) + _min_value_ozone
    return unnormalize_data

