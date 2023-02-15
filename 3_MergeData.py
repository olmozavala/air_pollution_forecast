from conf.localConstants import constants

from conf.TrainingUserConfiguration import getMergeParams
from conf.params import LocalTrainingParams, MergeFilesParams
from ai_common.constants.AI_params import TrainingParams
from AI.utils import getQuadrantsAsString
from datetime import date, datetime, timedelta
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
    print("Filtering dates with available meteorological dates....")
    not_meteo_idxs = []
    dates = [cur_datetime.date() for cur_datetime in datetimes]
    required_days = int(np.ceil( (forecasted_hours+1)/num_hours_in_netcdf)) # Number of days required for each date

    for date_idx, cur_datetime in enumerate(datetimes[:-required_days]):
        for i in range(required_days):
            cur_date_str = date.strftime(dates[date_idx+i], constants.date_format.value)
            required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")
            # Verify the desired file exist (if not it means we don't have meteo data for that pollution variable)
            if not (os.path.exists(required_netCDF_file_name)):
                # print(f"Warning! Meteorological file not found: {required_netCDF_file_name}") # For debugging  purposes
                # Reads the proper netcdf file
                not_meteo_idxs.append(date_idx)

    return not_meteo_idxs

def getMeteoColumns(file_name, forecasted_hours):
    """Simple function to get the meteorologica columns from the netcdf and create the 'merged ones' for all
    the 24 or 'forecasted_hours' """
    netcdf_data = pd.read_csv(file_name, index_col=0)
    meteo_columns = netcdf_data.axes[1].values
    all_meteo_columns = []
    for cur_forcasted_hour in range(forecasted_hours):
        for cur_col in meteo_columns:
            all_meteo_columns.append(F"{cur_col}_h{cur_forcasted_hour}")
    return meteo_columns, all_meteo_columns

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
    file_name = join(WRF_data_folder_name, F"{date.strftime(datetimes[0], constants.date_format.value)}.csv")
    meteo_columns, all_meteo_columns = getMeteoColumns(file_name, forecasted_hours) # Creates the meteo columns in the dataframe
    tot_meteo_columns = len(meteo_columns)
    x_data_meteo = np.zeros((tot_examples, tot_meteo_columns * forecasted_hours))
    rainc_cols = [x for x in meteo_columns if x.find('RAINC') != -1]
    rainnc_cols = [x for x in meteo_columns if x.find('RAINNC') != -1]
    tot_cols_per_row = tot_meteo_columns * forecasted_hours

    loaded_files = []  # A list off files that have been loaded already
    for date_idx, cur_datetime in enumerate(datetimes):
        if date_idx % (24*30) == 0:
            print(cur_datetime)

        # The + 1 is required to process variables like RAINC which needs the next hour
        required_days_to_read = int(np.ceil((forecasted_hours+1)/num_hours_in_netcdf))
        required_files = []
        files_available = True
        for day_idx in range(required_days_to_read):
            cur_date_str = date.strftime(datetimes[date_idx] + timedelta(days=day_idx), constants.date_format.value)
            netcdf_file = join(WRF_data_folder_name, F"{cur_date_str}.csv")
            if not(os.path.exists(netcdf_file)):
                files_available = False
                break
            else:
                required_files.append(netcdf_file)

        if not(files_available):
            break

        # Loading all the required files for this date
        files_not_loaded = [x for x in required_files if x not in loaded_files]

        if len(files_not_loaded) > 0:
            loaded_files = []  # clear the list of loaded files
            for file_idx, cur_file in enumerate(required_files):
                if len(loaded_files) == 0:  # Only when we don't have any useful file already loaded
                    netcdf_data = pd.read_csv(cur_file, index_col=0)
                else:
                    # Remove all dates
                    netcdf_data = pd.concat([netcdf_data,pd.read_csv(cur_file, index_col=0)])
                loaded_files.append(cur_file)

            # --------------------- Preprocess RAINC and RAINNC--------------------
            netcdf_data.iloc[:-1, [netcdf_data.columns.get_loc(x) for x in rainc_cols]] = netcdf_data.iloc[1:][rainc_cols].values - netcdf_data.iloc[:-1][rainc_cols].values
            netcdf_data.iloc[:-1, [netcdf_data.columns.get_loc(x) for x in rainnc_cols]] = netcdf_data.iloc[1:][rainnc_cols].values - netcdf_data.iloc[:-1][rainnc_cols].values
            # The last day between the years gets messed up, fix it by setting rain to 0
            netcdf_data[rainc_cols].where(netcdf_data[rainc_cols] <= 0, 0)
            netcdf_data[rainnc_cols].where(netcdf_data[rainnc_cols] <= 0, 0)
            np_flatten_data = netcdf_data.values.flatten()

        cur_hour = datetimes[date_idx].hour
        start_idx = cur_hour * tot_meteo_columns  # First column to copy from the current day
        end_idx = start_idx + tot_cols_per_row
        # print(F"{start_idx} - {end_idx}")
        x_data_meteo[date_idx, :] = np_flatten_data[start_idx:end_idx]

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

            print(F"\t\tFiltering dates for the year {current_year}")
            start_date = F'{current_year}-01-01'
            end_date = F'{current_year+1}-01-01'
            datetimes = pd.to_datetime(data_pollutants.index)
            not_valid_dates = np.logical_not((datetimes >= np.datetime64(start_date)) & (datetimes < np.datetime64(end_date)))
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
            cur_output_folder = join(output_folder, f"{num_quadrants}")
            if not(os.path.exists(cur_output_folder)):
                os.makedirs(cur_output_folder)
            x_data_merged_df.to_csv(join(cur_output_folder, F"{current_year}_{output_file_name}"),
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
                print(F"============ Pollutant: {cur_pollutant} -- Station:  {cur_station} ==========================")

                print(F"\tReading pollutant data: {cur_pollutant}_{cur_station}.csv ...")
                db_file_name = join(input_folder, constants.db_output_folder.value, F"{cur_pollutant}_{cur_station}.csv")
                data_cur_station = pd.read_csv(db_file_name, index_col=0)
                print("\tDone!")

                # Build the X and Y values for the training
                datetimes_str = data_cur_station.index.values
                datetimes = [datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str]
                tot_examples = len(datetimes_str)

                # Refresh valid dates
                print(F"\tTotal examples: {tot_examples}. Dates from {datetimes_str[0]} to {datetimes_str[-1]}")
                not_meteo_idxs = filterDatesWithMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf, WRF_data_folder_name)
                print(f"\t Times with no meteorological data: {[datetimes_str[x] for x in not_meteo_idxs]}")

                # Remove pollutant data where we don't have meteorolical data (removed from training examples)
                data_cur_station = data_cur_station.drop([datetimes_str[x] for x in not_meteo_idxs])

                print(F"\tReading meteorological data: {WRF_data_folder_name} ...")
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

