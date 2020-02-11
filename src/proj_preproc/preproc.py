import numpy as np
from datetime import datetime, timedelta

from conf.localConstants import constants

# Manually setting the min/max values for the pollutant (ozone)
_min_value_ozone = 0
_max_value_ozone = 150

def normalizeAndFilterData(data, datetimes_str, forecasted_hours):
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
    datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

    stations_columns = [x for x in data.columns.values if x.find('h') == -1]
    meteo_columns = [x for x in data.columns.values if x.find('h') != -1]

    # Normalizing meteorological variables
    min_values_meteo = data[meteo_columns].min()
    max_values_meteo = data[meteo_columns].max()


    data_norm_df = data.copy()
    data_norm_df[meteo_columns] = (data_norm_df[meteo_columns] - min_values_meteo)/(max_values_meteo - min_values_meteo)
    data_norm_df[stations_columns] = (data_norm_df[stations_columns] - _min_value_ozone)/(_max_value_ozone - _min_value_ozone)
    print(F'Done!')

    # Filtering only dates where there is data "forecasted hours after" (24 hrs after)
    print(F"\tBuilding X and Y ....")
    accepted_times_idx = []
    y_times_idx = []
    for i, c_datetime in enumerate(datetimes):
        forecasted_datetime = (c_datetime + timedelta(hours=forecasted_hours))
        if forecasted_datetime in datetimes:
            accepted_times_idx.append(i)
            y_times_idx.append(np.argwhere(forecasted_datetime == datetimes)[0][0])

    # Replacing nan columns with the mean value of all the other columns
    mean_values = data_norm_df[stations_columns].mean(1)

    data_norm_df_final = data_norm_df.copy()
    for cur_station in stations_columns:
        data_norm_df_final[cur_station] = data_norm_df[cur_station].fillna(mean_values)

    # print(F"Norm params: {scaler.get_params()}")
    # file_name_normparams = join(parameters_folder, F'{model_name}.txt')
    # utilsNN.save_norm_params(file_name_normparams, NormParams.min_max, scaler)

    return data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns

def deNormalize(data):
    unnormalize_data = data*(_max_value_ozone- _min_value_ozone) + _min_value_ozone
    return unnormalize_data
