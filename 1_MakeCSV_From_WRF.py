from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams
from conf.params import PreprocParams
from proj_preproc.wrf import crop_variables_xr, crop_variables_xr_cca_reanalisis, subsampleData
from proj_preproc.utils import getStringDates
from conf.localConstants import constants
from conf.localConstants import wrfFileType
import os
from os.path import join
import xarray as xr
# from img_viz.eoa_viz import EOAImageVisualizer
from multiprocessing import Pool

from proj_io.inout import read_wrf_files_names, read_wrf_old_files_names, saveFlattenedVariables

def process_files(user_config, all_path_names, all_file_names, all_dates, all_files_coords_old=[], mode=wrfFileType.new):
    variable_names = user_config[PreprocParams.variables]
    output_folder = user_config[PreprocParams.output_folder]
    output_folder_imgs= user_config[PreprocParams.output_imgs_folder]
    output_sizes = user_config[PreprocParams.resampled_output_sizes]
    bbox = user_config[PreprocParams.bbox]
    times = user_config[PreprocParams.times]

    # viz_obj = EOAImageVisualizer(output_folder=output_folder_imgs, disp_images=False)
    # Itereate over each file and preprocess them
    print("Processing new model files...")
    for file_idx in range(len(all_path_names)):
        print(F"================ {all_file_names[file_idx]} ================================ ")
        # Read file as xarray
        cur_xr_ds = xr.open_dataset(all_path_names[file_idx], decode_times=False)
        # Printing the summary of the data
        # viz_obj.xr_summary(cur_xr_ds)
        # print(F"\tCropping...")
        # Crops the desired variable_names
        try:
            if mode == wrfFileType.new:
                # In this case we have more than 48 hrs in each file
                cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, variable_names, bbox, times=times)

            if mode == wrfFileType.old:
                cur_xr_ds_coords = xr.open_dataset(all_files_coords_old[file_idx])
                LAT = cur_xr_ds_coords.XLAT.values[0,:,0]
                LON = cur_xr_ds_coords.XLONG.values[0,0,:]
                times = range(24) # These files only have 24 timesj
                cropped_xr_ds, newLAT, newLon = crop_variables_xr_cca_reanalisis(cur_xr_ds, variable_names, bbox,
                                                                                 times=times, LAT=LAT, LON=LON)
        except Exception as e:
            print(F"ERROR!!!!! Failed to crop file {all_path_names[file_idx]}: {e}")
            continue

        # viz_obj.xr_summary(cropped_xr_ds)
        # print("\tDone!")

        # For debugging, visualizing results
        # print("\tVisualizing cropped results...")
        # file_text = F"{all_file_names[file_idx]}"
        # viz_obj.plot_3d_data_xarray_map(cur_xr_ds, var_names=[variable_names[0]],
        #                                  timesteps=[0], title='Original Data', file_name_prefix='Original_{file_text}', timevar_name='time')
        # viz_obj.plot_3d_data_xarray_map(cropped_xr_ds, var_names=variable_names, timesteps=[0, 1], title='Cropped Data',
        #                                 file_name_prefix=F'Cropped_{file_text}', timevar_name='newtime')

        for output_size in output_sizes:
            output_folder_final = F"{output_folder}_{output_size['rows']}_{output_size['cols']}"
            if not (os.path.exists(output_folder_final)):
                os.makedirs(output_folder_final)
            # Subsample the data
            # print(F"\tSubsampling...")

            try:
                subsampled_xr_ds = subsampleData(cropped_xr_ds, variable_names, output_size['rows'], output_size['cols'])
            except Exception as e:
                print(F"ERROR!!!!! Failed to subsample file {all_path_names[file_idx]}, output size: {output_size}: {e}")
                continue
            # viz_obj.xr_summary(subsampled_xr_ds)
            # print("\tDone!")

            # For debugging
            # print("\tVisualizing subsampled results...")
            # file_text = F"{output_size['rows']}x{output_size['cols']}_{all_file_names[file_idx]}"
            # viz_obj.plot_3d_data_xarray_map(subsampled_xr_ds, var_names=variable_names, timesteps=[0,1], title='Subsampled Data',
            #                                 file_name_prefix=F"Subsampled_{file_text}", timevar_name='newtime')

            # print(f"\tFlattening variables and saving as csv {join(output_folder_final, all_dates[file_idx].strftime(constants.date_format.value))}")
            # Obtain time strings for current file
            # Save variables as a single CSV file

            try:
                saveFlattenedVariables(subsampled_xr_ds, variable_names, output_folder_final,
                                       file_name=F"{all_dates[file_idx].strftime(constants.date_format.value)}.csv",
                                       index_names=getStringDates(all_dates[file_idx], times),
                                       index_label=constants.index_label.value)
            except Exception as e:
                print(F"ERROR!!!!! Failed with file {all_path_names[file_idx]}: {e}")
            continue

def runParallel(year):
    user_config = getPreprocWRFParams()

    input_folder = user_config[PreprocParams.input_folder_new]
    input_folder_old = user_config[PreprocParams.input_folder_old]

    start_date = F'{year}-01-01'
    end_date = F'{year + 1}-01-01'

    if year < 2018: # We use the 'old' model
        print(f"Working with old model files years {start_date}-{end_date}")
        all_dates_old, all_file_names_old, all_files_coords_old, all_path_names_old = read_wrf_old_files_names(
                        input_folder_old, start_date, end_date)
        process_files(user_config, all_path_names_old, all_file_names_old, all_dates_old, all_files_coords_old, mode=wrfFileType.old)
    else:
        print(f"Working with new model files years {start_date}-{end_date}")
        all_dates, all_file_names, all_path_names = read_wrf_files_names(input_folder, start_date, end_date)
        process_files(user_config, all_path_names, all_file_names, all_dates, [], mode=wrfFileType.new)
    print("Done!")


if __name__== '__main__':
    # Reads user configuration
    user_config = getPreprocWRFParams()
    input_folder = user_config[PreprocParams.input_folder_new]
    input_folder_old = user_config[PreprocParams.input_folder_old]

    # The max range is from 1980 to present
    start_year = 2010
    end_year = 2023

    # Run this process in parallel splitting separating by years
    NUMBER_PROC = 20
    p = Pool(NUMBER_PROC)
    p.map(runParallel, range(start_year, end_year))