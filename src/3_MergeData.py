from conf.localConstants import constants

from conf.TrainingUserConfiguration import getMergeParams
from conf.params import LocalTrainingParams, MergeFilesParams
from training.utils import getQuadrantsAsString
from AI.data_generation.Generators3D import *
from datetime import date, datetime
import os
from inout.io_common import create_folder

import pandas as pd
from pandas import DataFrame

from constants.AI_params import *

if __name__ == '__main__':

    config = getMergeParams()

    input_folder = config[TrainingParams.input_folder]
    output_folder = config[MergeFilesParams.output_folder]
    start_date = config[LocalTrainingParams.start_date]
    end_date = config[LocalTrainingParams.end_date]
    stations = config[LocalTrainingParams.station]
    pollutants = config[LocalTrainingParams.pollutant]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    num_quadrants = config[LocalTrainingParams.tot_num_quadrants]
    num_hours_in_netcdf  = config[LocalTrainingParams.num_hours_in_netcdf]

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
                data_pollutant = pd.read_csv(db_file_name, index_col=0)
                print("\tDone!")

                # Build the X and Y values for the training
                datetimes_str = data_pollutant.index.values
                datetimes = [datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str]
                dates = [cur_datetime.date() for cur_datetime in datetimes]

                # Quick filter on which hours are really possible to use
                print(F"\tFiltering dates...")
                not_meteo_idxs = []
                for date_idx, cur_datetime in enumerate(datetimes):
                    cur_date_str = date.strftime(dates[date_idx], constants.date_format.value)
                    cur_hour = datetimes[date_idx].hour
                    # Obtain the name of the required file
                    if cur_hour+forecasted_hours < num_hours_in_netcdf:
                        required_netCDF_file_name = join(WRF_data_folder_name,F"{cur_date_str}.csv")
                    else:
                        required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")

                    # Verify the desired file exist (if not it means we don't have meteo data for that pollution variable)
                    if not(os.path.exists(required_netCDF_file_name)):
                        # Reads the proper netcdf file
                        not_meteo_idxs.append(date_idx)

                data_pollutant = data_pollutant.drop([datetimes_str[x] for x in not_meteo_idxs])
                # Refresh valid dates
                datetimes_str = data_pollutant.index.values
                tot_examples = len(datetimes_str)
                print(F"\tOriginal examples: {len(datetimes)} new examples: {tot_examples}")
                datetimes = [datetime.strptime(x, constants.datetime_format.value) for x in data_pollutant.index.values]
                dates = [cur_datetime.date() for cur_datetime in datetimes]

                # To make it more efficient we verify which netcdf data was loaded previously
                prev_loaded_file = ''
                print(F"\tAppending meteorological data...")
                for date_idx, cur_datetime in enumerate(datetimes):
                    cur_date_str = date.strftime(dates[date_idx], constants.date_format.value)
                    cur_hour = datetimes[date_idx].hour
                    # Obtain the name of the required file
                    if cur_hour+forecasted_hours < num_hours_in_netcdf:
                        required_netCDF_file_name = join(WRF_data_folder_name,F"{cur_date_str}.csv")
                    else:
                        cur_date_str = date.strftime(dates[date_idx+1], constants.date_format.value)
                        required_netCDF_file_name = join(WRF_data_folder_name, F"{cur_date_str}.csv")

                    # Verify the desired file exist (if not it means we don't have meteo data for that pollution variable)
                    if os.path.exists(required_netCDF_file_name):
                        # Reads the proper netcdf file
                        if required_netCDF_file_name != prev_loaded_file:
                            # print(F"\t\t Reading {required_netCDF_file_name}...")
                            prev_loaded_file = required_netCDF_file_name
                            netcdf_data = pd.read_csv(required_netCDF_file_name, index_col=0)
                            meteo_data = netcdf_data.values.flatten()

                            # Adding additional columns into the final dataframe, not sure if there is a faster way
                            if append_meteo_colums:
                                meteo_columns = netcdf_data.axes[1].values
                                tot_meteo_columns = len(meteo_columns)
                                x_data_meteo = np.zeros((tot_examples, tot_meteo_columns*forecasted_hours))
                                all_meteo_columns = []
                                for cur_forcasted_hour in range(forecasted_hours):
                                    for cur_col in meteo_columns:
                                        all_meteo_columns.append(F"{cur_col}_h{cur_forcasted_hour}")
                                append_meteo_colums = False

                        # print(F"\t\t\tAppending meteo data for current datetime: {datetimes[date_idx]}")
                        # Appends the meteo data into the proper row of the database
                        x_data_meteo[date_idx,:] = meteo_data[cur_hour*tot_meteo_columns:(cur_hour+forecasted_hours)*tot_meteo_columns]

                x_data_merged_df = DataFrame(x_data_meteo, columns=all_meteo_columns, index=datetimes_str)
                x_data_merged_df[cur_pollutant] = data_pollutant[cur_pollutant]
                print("\tSaving merged database ...")
                output_file_name = F"{cur_pollutant}_{cur_station}.csv"
                x_data_merged_df.to_csv(join(output_folder,output_file_name),
                                        float_format="%.4f",
                                        index_label = constants.index_label.value)
                print("\tDone!")
            except Exception as e:
                print(F"ERROR!!!!! It failed for {cur_pollutant} -- {cur_station}: {e} ")

