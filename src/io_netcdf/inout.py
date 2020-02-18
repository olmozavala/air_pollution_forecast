from datetime import datetime, timedelta
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

        # Hardcoded variables to change the
        if cur_var_name in ['RAINC', 'RAINNC']:
            for i in range(times-1):
                cur_var_np[i, :, :] = cur_var_np[i+1, :, :] - cur_var_np[i, :, :]
        var_flat_values = np.array([cur_var_np[i,:,:].flatten() for i in range(times)])
        var_columns = [F"{cur_var_name}_{i}" for i in range(rows*cols)]
        temp_dict = {var_columns[i]: var_flat_values[:,i] for i in range(len(var_columns))}
        all_data = pd.concat([all_data, pd.DataFrame(temp_dict)], axis=1)

    all_data.set_axis(index_names, inplace=True)
    all_data.to_csv(join(output_folder,file_name), index_label=index_label);

