from datetime import datetime, timedelta, date
from conf.localConstants import constants
from os.path import join
import numpy as np
import pandas as pd
import os
import re

def read_wrf_old_files_names(input_folder, start_date, end_date):
    """
    Function to save the address of the netCDF in a txt file

    :param input_folder: address to copy the file
    :type input_folder: String
    :param pathNetCDF: address where the xr_ds files are located
    :type pathNetCDF: String
    """
    start_date = datetime.strptime(start_date, constants.date_format.value)
    end_date = datetime.strptime(end_date, constants.date_format.value)
    input_folder
    name_pattern = 'wrfout_c1h_d01_\d\d\d\d-\d\d-\d\d_00:00:00.a\d\d\d\d'
    # name_pattern = 'wrfout_c1h_d01_\d\d\d\d-\d\d-\d\d_00:00:00.\d\d\d\d'
    date_pattern = '\d\d\d\d-\d\d-\d\d'
    file_re = re.compile(name_pattern + '.*')
    date_re = re.compile(date_pattern)

    result_files = []
    result_files_coords = []
    result_paths = []
    result_dates = []
    # Iterate over the years
    for cur_year in range(start_date.year, end_date.year+1):
        all_files = os.listdir(join(input_folder, F"a{cur_year}", 'salidas'))
        # all_files = os.listdir(join(input_folder, F"a{cur_year}"))
        # Get all domain files (we have two domains now)
        all_domain_files = [x for x in all_files if file_re.match(x) != None]
        all_domain_files.sort()
        # print(all_domain_files)
        # Verify the files are withing the desired dates
        for curr_file in all_domain_files:
            dateNetCDF = datetime.strptime(date_re.findall(curr_file)[0], '%Y-%m-%d')
            if (dateNetCDF < end_date) & (dateNetCDF >= start_date):
                result_files_coords.append(join(input_folder,F"a{cur_year}", 'salidas',
                            F'wrfout_c15d_d01_{cur_year}-01-01_00:00:00.a{cur_year}'))  # always read the first of jan (assuming it exist)
                # result_files_coords.append(join(input_folder,F"a{cur_year}",
                #                                 F'wrfout_c15d_d01_{cur_year}-01-01_00:00:00.{cur_year}'))  # always read the first of jan (assuming it exist)
                result_files.append(curr_file)
                result_paths.append(join(input_folder, F"a{cur_year}", 'salidas', curr_file))
                # result_paths.append(join(input_folder, F"a{cur_year}", curr_file))
                result_dates.append(dateNetCDF)
                print(F'{curr_file} -- {dateNetCDF}')

    return result_dates, result_files, result_files_coords, result_paths

def read_wrf_files_names(input_folder, start_date, end_date):
    """
    Function to save the address of the netCDF in a txt file

    :param input_folder: address to copy the file
    :type input_folder: String
    :param pathNetCDF: address where the xr_ds files are located
    :type pathNetCDF: String
    """
    # Check that the type is a string for start_date
    if type(start_date) is str:
        start_date = datetime.strptime(start_date, constants.date_format.value)
        end_date = datetime.strptime(end_date, constants.date_format.value)

    input_folder
    name_pattern = 'wrfout_d02_\d\d\d\d-\d\d-\d\d_00.nc'
    date_pattern = '\d\d\d\d-\d\d-\d\d'
    file_re = re.compile(name_pattern + '.*')
    date_re = re.compile(date_pattern)

    result_files = []
    result_paths = []
    result_dates = []
    # Iterate over the years
    for cur_year in range(start_date.year, end_date.year+1):
        months_in_year = os.listdir(join(input_folder, str(cur_year)))
        # Iterate over the months inside that year
        for cur_month in months_in_year:
            all_files = os.listdir(join(input_folder, str(cur_year), str(cur_month)))
            # Get all domain files (we have two domains now)
            all_domain_files = [x for x in all_files if file_re.match(x) != None]
            all_domain_files.sort()
            # print(all_domain_files)
            # Verify the files are withing the desired dates
            for curr_file in all_domain_files:
                dateNetCDF = datetime.strptime(date_re.findall(curr_file)[0], '%Y-%m-%d')
                if (dateNetCDF < end_date) & (dateNetCDF >= start_date):
                    result_files.append(curr_file)
                    result_paths.append(join(input_folder, str(cur_year), str(cur_month), curr_file))
                    result_dates.append(dateNetCDF)
                    print(F'{curr_file} -- {dateNetCDF}')

    return result_dates, result_files, result_paths

def saveFlattenedVariables(xr_ds, variable_names, output_folder, file_name, index_names, index_label=''):
    """ This function saves the data in a csv file format. It generates a single column for each
    value of each variable, and one row for each time step"""
    all_data = pd.DataFrame()
    for cur_var_name in variable_names:
        cur_var = xr_ds[cur_var_name]
        cur_var_np = xr_ds[cur_var_name].values
        size_defined = False
        dims = cur_var.shape

        # TODO hardcoded order of dimensions
        times = dims[0]
        rows = dims[1]
        cols = dims[2]

        var_flat_values = np.array([cur_var_np[i,:,:].flatten() for i in range(times)])
        var_columns = [F"{cur_var_name}_{i}" for i in range(rows*cols)]
        temp_dict = {var_columns[i]: var_flat_values[:,i] for i in range(len(var_columns))}
        all_data = pd.concat([all_data, pd.DataFrame(temp_dict)], axis=1)

    all_data.set_axis(index_names, inplace=True)
    all_data.to_csv(join(output_folder,file_name), index_label=index_label, float_format = "%.4f")

def readMeteorologicalData(datetimes, forecasted_hours, num_hours_in_netcdf, WRF_data_folder_name):
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
    print("Reading meteorological data...")
    tot_examples = len(datetimes)
    file_name = join(WRF_data_folder_name, F"{date.strftime(datetimes[0], constants.date_format.value)}.csv")
    meteo_columns, all_meteo_columns = getMeteoColumns(file_name, forecasted_hours) # Creates the meteo columns in the dataframe
    tot_meteo_columns = len(meteo_columns)
    x_data_meteo = np.zeros((tot_examples, tot_meteo_columns * forecasted_hours))
    rainc_cols = [x for x in meteo_columns if x.find('RAINC') != -1]
    rainnc_cols = [x for x in meteo_columns if x.find('RAINNC') != -1]
    tot_cols_per_row = tot_meteo_columns * forecasted_hours

    loaded_files = []  # A list off files that have been loaded already (to make it efficient)
    # Load by date
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
            print(f"WARNING: The required files are not available for date {cur_datetime}")
            continue

        # Loading all the required files for this date (checking it has not been loaded before)
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

def generateDateColumns(datetimes):
    time_cols = [ 'half_sin_day', 'half_cos_day', 'half_sin_year', 'half_cos_year', 'half_sin_week', 'half_cos_week',
                    'sin_day', 'cos_day', 'sin_year', 'cos_year', 'sin_week', 'cos_week']
    # Incorporate dates into the merged dataset sin and cosines
    day = 24 * 60 * 60
    week = day * 7
    year = (365.2425) * day

    two_pi = 2 * np.pi
    options = [day, week, year]

    time_values = []
    # Get the sin and cos for each of the options for half the day
    for c_option in options:
        time_values.append(np.array([np.abs(np.sin(x.timestamp() * (np.pi / c_option))) for x in datetimes]))
        time_values.append(np.array([np.abs(np.cos(x.timestamp() * (np.pi / c_option))) for x in datetimes]))

    # Get the sin and cos for each of the options
    for c_option in options:
        time_values.append(np.array([np.sin(x.timestamp() * (two_pi / c_option)) for x in datetimes]))
        time_values.append(np.array([np.cos(x.timestamp() * (two_pi / c_option)) for x in datetimes]))

    # Plot obtained vlaues
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,1, figsize=(15,5))
    # for i, curr_time_var in enumerate(time_values):
    #     ax.plot(curr_time_var[:24*7], label=time_cols[i])
    # ax.legend()
    # plt.savefig('test.png')
    # plt.close()

    return time_cols, time_values

def read_merged_files(input_folder, start_year, end_year):
    # -------- Reading all the years in a single data frame (all stations)
    print(f"Reading years {start_year} to {end_year}...")
    for c_year in range(start_year, end_year+1):
        db_file_name = join(input_folder, F"{c_year}_AllStations.csv") # Just for testing
        print(F"============ Reading data for: {c_year}: {db_file_name}")
        if c_year == start_year:
            data = pd.read_csv(db_file_name, index_col=0)
        else:
            data = pd.concat([data, pd.read_csv(db_file_name, index_col=0)])

    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")
    return data

def get_column_names(df):
    '''
    Reads all the column nams of the specified dataframe. It separates the names for pollution,
    meteorological, and time columns
    '''
    myregex = f"cont_.*"
    all_contaminant_columns = df.filter(regex=myregex).columns
    # print(all_contaminant_columns.values)
    all_time_colums = df.filter(regex="day|year|week").columns
    # print(all_time_colums.values)
    all_meteo_columns = [x for x in df.columns if x not in all_contaminant_columns and x not in all_time_colums]
    # print(all_meteo_columns)
    return all_contaminant_columns, all_meteo_columns, all_time_colums

def filter_data(df, filter_type='none', filtered_pollutant='', filtered_station=''):
    '''
    This code can filter the dataframe by the specified filter type. 
    The possible filter_types are: 'none', 'single_pollutant', 'single_pollutant_and_station'
    '''
    all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(df)
    if filter_type == 'single_pollutant': # In case we want to use a single pollutant and station
        # ---------- Here we only keep the columns for the current pollutant all stations
        keep_cols = [x for x in df.columns if x.startswith(f'cont_{filtered_pollutant}')] + all_time_colums.tolist() + all_meteo_columns
        print(F"Keeping columns: {len(keep_cols)} original columns: {len(df.columns)}")
        X_df = df[keep_cols].copy()
    elif filter_type == 'single_pollutant_and_station': # In case we want to use a single pollutant and station
        # ------------- Here we only keep the columns for the current station and pollutant
        keep_cols = [f'cont_{filtered_pollutant}_{filtered_station}'] + all_time_colums.tolist() + all_meteo_columns
        print(F"Keeping columns: {len(keep_cols)} original columns: {len(df.columns)}")
        X_df = df[keep_cols].copy()
    elif filter_type == 'none':
        X_df = df.copy()

    return X_df

def add_previous_hours(df, hours_before=24):
    '''
    This function adds the previous hours of the pollutants as additional columns
    '''
    print("\tAdding the previous hours of the pollutants as additional columns...")

    contaminant_columns, _, _= get_column_names(df)
    print(F"\t\tContaminant columns: {contaminant_columns.values}")
    for c_hour in range(1, hours_before+1):
        for c_column in contaminant_columns:
            df[f'minus_{c_hour:02d}_{c_column}'] = df[c_column].shift(c_hour)
    print(F'X {df.shape}, Memory usage: {df.memory_usage().sum()/1024**2:02f} MB')
    print("Done!")
    return df

def add_forecasted_hours(df, pollutant, forecasted_hours=range(1,25)):
    '''
    This function adds the forecasted hours of a single pollutant in a new dataframe
    forecasted_hours: Array with the hours to forecast
    '''
    myregex = f"^cont_{pollutant}.*"
    single_cont_columns = df.filter(regex=myregex).columns
    # print(single_cont_columns)

    # Adds the next 24 (forecasted_hours) hours to the prediction
    Y_df =  pd.DataFrame(index=df.index)
    for c_hour in forecasted_hours:
        for c_column in single_cont_columns:
            Y_df[f'plus_{c_hour:02d}_{c_column}'] = df[c_column].shift(-c_hour)

    print(f"Shape of Y: {Y_df.shape}")
    print("Done!")
    return Y_df

def save_columns(df, file_name):
    '''
    This function saves the columns of a dataframe to a csv file
    '''
    cols = pd.DataFrame(df.columns)
    cols.to_csv(file_name, index=False)
    print(f"Done saving file: {file_name}")