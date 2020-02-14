from conf.localConstants import constants

from conf.TrainingUserConfiguration import getMergeParams
from conf.params import LocalTrainingParams, MergeFilesParams
from constants.AI_params import TrainingParams
from AI.utils import getQuadrantsAsString
from datetime import date, datetime
from os.path import join
import os
import numpy as np

import pandas as pd
from pandas import DataFrame


def filterDatesWithMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf, WRF_data_folder_name):
    """
    Searches current datetimes in the netcdf list of files, verify
    :param datetimes:
    :param dates:
    :param forecasted_hours:
    :param num_hours_in_netcdf:
    :param WRF_data_folder_name:
    :return:
    """
    print(F"\tFiltering dates...")
    not_meteo_idxs = []
    dates = [cur_datetime.date() for cur_datetime in datetimes]
    for date_idx, cur_datetime in enumerate(datetimes):
        cur_date_str = date.strftime(dates[date_idx], constants.date_format.value)
        cur_hour = datetimes[date_idx].hour
        # Obtain the name of the required file
        if cur_hour + forecasted_hours < num_hours_in_netcdf:
            required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")
        else:
            # TODO here we should get the next day forecast file
            required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")

        # Verify the desired file exist (if not it means we don't have meteo data for that pollution variable)
        if not (os.path.exists(required_netCDF_file_name)):
            # Reads the proper netcdf file
            not_meteo_idxs.append(date_idx)

    return not_meteo_idxs


def readMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf, WRF_data_folder_name, tot_examples):
    """
    Reads the meteorological variables for the desired datetimes. It is assumed that the file exists
    :param datetimes:
    :param forecasted_hours:
    :param num_hours_in_netcdf:
    :param WRF_data_folder_name:
    :param tot_examples:
    :return:
    """
    # To make it more efficient we verify which netcdf data was loaded previously
    dates = [cur_datetime.date() for cur_datetime in datetimes]
    prev_loaded_file = ''
    next_day_prev_loaded_file = ''
    append_meteo_colums = True
    x_data_meteo = None
    all_meteo_columns = None
    print(F"\tAppending meteorological data...")
    for date_idx, cur_datetime in enumerate(datetimes):
        if date_idx % (24*365) == 0:
            print(cur_datetime)

        cur_date_str = date.strftime(dates[date_idx], constants.date_format.value)
        cur_hour = datetimes[date_idx].hour

        # Obtain the name of the necdf file or files (if we need data from two files)
        required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")
        next_day_netCDF_file_name = required_netCDF_file_name
        if cur_hour + forecasted_hours > num_hours_in_netcdf:
            if (date_idx + 1) < len(datetimes): # Simple validation of the last date
                cur_date_str = date.strftime(datetimes[date_idx] + np.timedelta64(1, 'D'), constants.date_format.value)
                next_day_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")
            else:
                continue

        # Verify the desired file exist (if not it means we don't have meteo data for that pollution variable)
        if os.path.exists(required_netCDF_file_name) & os.path.exists(next_day_netCDF_file_name):
            # Reads the proper netcdf file
            if required_netCDF_file_name != prev_loaded_file:
                # print(F"\t\t Reading {required_netCDF_file_name}...")
                prev_loaded_file = required_netCDF_file_name
                netcdf_data = pd.read_csv(required_netCDF_file_name, index_col=0)
                meteo_data = netcdf_data.values.flatten()

            # === TODO this is executed only once, move outsite the loop Adding additional columns into the final dataframe, not sure if there is a faster way
            if append_meteo_colums:
                meteo_columns = netcdf_data.axes[1].values
                tot_meteo_columns = len(meteo_columns)
                x_data_meteo = np.zeros((tot_examples, tot_meteo_columns * forecasted_hours))
                all_meteo_columns = []
                for cur_forcasted_hour in range(forecasted_hours):
                    for cur_col in meteo_columns:
                        all_meteo_columns.append(F"{cur_col}_h{cur_forcasted_hour}")
                append_meteo_colums = False

            if cur_hour + forecasted_hours > num_hours_in_netcdf:
                if next_day_prev_loaded_file != next_day_netCDF_file_name:
                    next_day_prev_loaded_file = next_day_netCDF_file_name
                    next_day_netcdf_data = pd.read_csv(next_day_netCDF_file_name, index_col=0)
                    next_day_meteo_data = next_day_netcdf_data.values.flatten()

                # Appends the meteo data into the proper row of the database
                start_idx = cur_hour * tot_meteo_columns  # First column to copy from the current day
                end_idx = tot_meteo_columns * (forecasted_hours - cur_hour)
                x_data_meteo[date_idx, :end_idx] = meteo_data[start_idx:]
                start_idx = end_idx
                end_idx = (num_hours_in_netcdf - forecasted_hours + cur_hour) * tot_meteo_columns  # Last column to copy from the next day
                x_data_meteo[date_idx, start_idx:] = next_day_meteo_data[:end_idx]
            else:
                x_data_meteo[date_idx, :] = meteo_data

    return x_data_meteo, all_meteo_columns


def merge_by_year(config):
    input_folder = config[MergeFilesParams.input_folder]
    output_folder = config[MergeFilesParams.output_folder]
    stations = config[MergeFilesParams.stations]
    pollutants = config[MergeFilesParams.pollutant_tables]
    forecasted_hours = config[MergeFilesParams.forecasted_hours]
    num_quadrants = config[LocalTrainingParams.tot_num_quadrants]
    num_hours_in_netcdf = config[LocalTrainingParams.num_hours_in_netcdf]
    years = config[MergeFilesParams.years]

    WRF_data_folder_name = join(input_folder, constants.wrf_output_folder.value,
                                F"{constants.wrf_each_quadrant_name.value}_{getQuadrantsAsString(num_quadrants)}")

    if not (os.path.exists(output_folder)):
        os.makedirs(output_folder)

    # =============== Read data and merge meteorological variables===============
    data_pollutants = None

    # Iterate over all pollutants
    for cur_pollutant in pollutants:
        # Obtain all the 'available pollution dates'
        print(F"\tReading data...")
        notfound = []
        for cur_station in stations:
            print(F"============  {cur_pollutant} -- {cur_station} ==========================")
            db_file_name = join(input_folder, constants.db_output_folder.value,
                                F"{cur_pollutant}_{cur_station}.csv")

            if not (os.path.exists(db_file_name)):
                notfound.append(cur_station)
                continue

            # These are integer values so we can make the reading more efficiently
            if cur_pollutant in ['cont_otres']:
                data_cur_station = pd.read_csv(db_file_name,  index_col=0, parse_dates=True, dtype={cur_pollutant: np.int32})
            else:
                data_cur_station = pd.read_csv(db_file_name, index_col=0, parse_dates=True, )

            data_cur_station = data_cur_station.rename(columns={cur_pollutant: cur_station})

            if data_pollutants is None:
                data_pollutants = data_cur_station
            else:
                data_pollutants = pd.concat([data_pollutants, data_cur_station], axis=1)

        # Iterate over all the years
        for current_year in years:
            print(F"\tDone!  Not found: {notfound}")

            print(F"Filtering dates for the year {current_year}")
            start_date = F'{current_year}-01-01'
            # end_date = F'{current_year}-01-05'
            end_date = F'{current_year+1}-01-01'
            datetimes = pd.to_datetime(data_pollutants.index)
            not_valid_dates = np.logical_not( (datetimes >= np.datetime64(start_date)) & (datetimes < np.datetime64(end_date)))
            data_pollutants_filtered = data_pollutants.drop(data_pollutants.index[not_valid_dates])
            # Reloading the dates
            datetimes = pd.to_datetime(data_pollutants_filtered.index)

            # Filtering dates that are not available in the meteorological data
            print(F"Filtering dates with meteorological information")
            not_meteo_idxs = filterDatesWithMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf,
                                                               WRF_data_folder_name)

            # Remove pollutant data where we don't have meteorolical data (removed from training examples)
            data_pollutants_filtered = data_pollutants_filtered.drop([datetimes[x] for x in not_meteo_idxs])

            # Refresh valid dates
            tot_examples = len(data_pollutants_filtered.index)
            print(F"\tOriginal examples: {len(datetimes)} new examples: {tot_examples}")
            datetimes = pd.to_datetime(data_pollutants_filtered.index)

            x_data_meteo, all_meteo_columns = readMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf,
                                                                     WRF_data_folder_name, tot_examples)

            x_data_merged_df = DataFrame(x_data_meteo, columns=all_meteo_columns, index=data_pollutants_filtered.index)
            final_stations = data_pollutants_filtered.columns.values
            for cur_station in final_stations:
                x_data_merged_df[cur_station] = data_pollutants_filtered[cur_station]

            print("\tSaving merged database ...")
            output_file_name = F"{cur_pollutant}_AllStations.csv"
            x_data_merged_df.to_csv(join(output_folder,{date.today().strftime('%Y_%m_%d')}, F"{current_year}_{output_file_name}"),
                                    float_format="%.2f",
                                    index_label=constants.index_label.value)
            print("\tDone!")


def merge_by_station(config):
    input_folder = config[MergeFilesParams.input_folder]
    output_folder = config[MergeFilesParams.output_folder]
    stations = config[MergeFilesParams.stations]
    pollutants = config[MergeFilesParams.pollutant_tables]
    forecasted_hours = config[MergeFilesParams.forecasted_hours]
    num_quadrants = config[LocalTrainingParams.tot_num_quadrants]
    num_hours_in_netcdf = config[LocalTrainingParams.num_hours_in_netcdf]

    WRF_data_folder_name = join(input_folder,constants.wrf_output_folder.value,
                                F"{constants.wrf_each_quadrant_name.value}_{getQuadrantsAsString(num_quadrants)}")

    if not (os.path.exists(output_folder)):
        os.makedirs(output_folder)

    # =============== Read data and merge meteorological variables===============
    # Iterate over all pollutants
    for cur_pollutant in pollutants:

        # Iterate over all stations
        for cur_station in stations:
            try:
                print(F"============  {cur_pollutant} -- {cur_station} ==========================")

                print(F"\tReading data...")
                append_meteo_colums = True  # It is used to append the meteo variable columns into the database

                db_file_name = join(input_folder, constants.db_output_folder.value, F"{cur_pollutant}_{cur_station}.csv")
                data_cur_station = pd.read_csv(db_file_name, index_col=0)
                print("\tDone!")

                # Build the X and Y values for the training
                datetimes_str = data_cur_station.index.values
                datetimes = [datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str]

                # Quick filter on which hours are really possible to use

                # Refresh valid dates
                datetimes_str = data_cur_station.index.values
                tot_examples = len(datetimes_str)
                print(F"\tOriginal examples: {len(datetimes)} new examples: {tot_examples}")
                datetimes = [datetime.strptime(x, constants.datetime_format.value) for x in data_cur_station.index.values]

                not_meteo_idxs = filterDatesWithMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf, WRF_data_folder_name)

                # Remove pollutant data where we don't have meteorolical data (removed from training examples)
                data_cur_station = data_cur_station.drop([datetimes_str[x] for x in not_meteo_idxs])

                x_data_meteo, all_meteo_columns = readMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf,
                                                                         WRF_data_folder_name, tot_examples)

                x_data_merged_df = DataFrame(x_data_meteo, columns=all_meteo_columns, index=datetimes_str)
                x_data_merged_df[cur_pollutant] = data_cur_station[cur_pollutant]
                print("\tSaving merged database ...")
                output_file_name = F"{cur_pollutant}_{cur_station}.csv"
                x_data_merged_df.to_csv(join(output_folder,output_file_name),
                                        float_format="%.2f",
                                        index_label=constants.index_label.value)
                print("\tDone!")
            except Exception as e:
                print(F"ERROR!!!!! It failed for {cur_pollutant} -- {cur_station}: {e} ")


if __name__ == '__main__':

    config = getMergeParams()
    # merge_by_station(config)
    merge_by_year(config)

