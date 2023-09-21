from conf.localConstants import constants

from conf.TrainingUserConfiguration import getMergeParams
from conf.params import LocalTrainingParams, MergeFilesParams
from ai_common.constants.AI_params import TrainingParams
from AI.utils import getQuadrantsAsString
from datetime import date, datetime, timedelta
from os.path import join
import os
import numpy as np
from proj_io.inout import readMeteorologicalData, generateDateColumns, filterDatesWithMeteorologicalData

import pandas as pd
from pandas import DataFrame


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
    alldata_by_year = None

    # Iterate over all pollutants
    for current_year in years:
        # Obtain all the 'available pollution dates'
        print(F"\tReading data for year {current_year}...")
        notfound = []
        datetimes = pd.date_range(start=F'{current_year}-01-01', end= F'{current_year+1}-01-01', freq='H')
        alldata_by_year = pd.DataFrame(data=[], index=datetimes)
        # Iterate over all the pollutants and create a single df
        for cur_pollutant in pollutants:
        # for cur_pollutant in ['cont_co']:
            # This loop merges the data for all stations a single pollutant
            for cur_station in stations:
            # for cur_station in ['IZT']:
                db_file_name = join(input_folder, constants.db_output_folder.value,
                                    F"{cur_pollutant}_{cur_station}.csv")
                print(F"============  {cur_pollutant} -- {cur_station} -- {db_file_name }==========================")

                if not (os.path.exists(db_file_name)):
                    notfound.append(cur_station)
                    continue

                # Otres are integer values so we can make the reading more efficiently
                if cur_pollutant in ['cont_otres']:
                    data_cur_station = pd.read_csv(db_file_name,  index_col=0, parse_dates=True, dtype={cur_pollutant: np.int32})
                else:
                    data_cur_station = pd.read_csv(db_file_name, index_col=0, parse_dates=True, )

                data_cur_station = data_cur_station.rename(columns={cur_pollutant: f'{cur_pollutant}_{cur_station}'})

                # print(f'The shape of the cur_station is (before): {data_cur_station.shape}')
                # print(f'The shape of the df is (before): {alldata_by_year.shape}')
                alldata_by_year = alldata_by_year.join(data_cur_station, how='left')

                # print(f'The shape of the df is (after): {alldata_by_year.shape}')
                # mem_usage = data_cur_station.memory_usage(deep=True).sum()/ (1024 ** 3)
                # print(f'Total memory usage cur_station: {mem_usage:.2f} GB')
                # mem_usage = alldata_by_year.memory_usage(deep=True).sum()/ (1024 ** 3)
                # print(f'Total memory usage all: {mem_usage:.2f} GB')


        x_data_meteo, all_meteo_columns = readMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf,
                                                                    WRF_data_folder_name)

        # Initialize the merged dataset with the meteorological data (using the same index/dates than the pollutants)
        alldata_by_year_meteo = DataFrame(x_data_meteo, columns=all_meteo_columns, index=alldata_by_year.index)

        # --------- Merge meto with pollutants
        alldata_by_year = alldata_by_year.join(alldata_by_year_meteo)

        # # ---------- Add the times columns (sin_day, cos_day, sin_year, cos_year, sin_week, cos_week)
        time_cols, time_values = generateDateColumns(datetimes=datetimes)

        for idx, cur_time_col in enumerate(time_cols):
            alldata_by_year[cur_time_col] = time_values[idx]


        print("\tSaving merged database ...")
        output_file_name = F"{current_year}_AllStations.csv"
        cur_output_folder = join(output_folder, f"{num_quadrants}")
        if not(os.path.exists(cur_output_folder)):
            os.makedirs(cur_output_folder)
        alldata_by_year.to_csv(join(cur_output_folder, output_file_name),
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
