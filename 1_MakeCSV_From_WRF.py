# %%
import sys
import numpy as np
# We need to add the path to the root folder of the project
# sys.path.append('/home/olmozavala/air_pollution_forecast/eoas_pyutils')
sys.path.append('/home/olmozavala/CODE/air_pollution_forecast/eoas_pyutils')

from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams
from conf.params import PreprocParams
from proj_preproc.wrf import crop_variables_xr, crop_variables_xr_cca_reanalisis, subsampleData
from proj_preproc.utils import getStringDates
from conf.localConstants import constants
from conf.localConstants import wrfFileType
import os
from os.path import join
import xarray as xr
from viz_utils.eoa_viz import EOAImageVisualizer, BackgroundType
from multiprocessing import Pool
# %%

from proj_io.inout import read_wrf_files_names, read_wrf_old_files_names, saveFlattenedVariables

def process_files(user_config, all_path_names, all_file_names, all_dates, all_files_coords_old=[], mode=wrfFileType.new):
    variable_names = user_config[PreprocParams.variables]
    output_folder = user_config[PreprocParams.output_folder]
    output_folder_imgs= user_config[PreprocParams.output_imgs_folder]
    output_sizes = user_config[PreprocParams.resampled_output_sizes]
    bbox = user_config[PreprocParams.bbox]
    times = user_config[PreprocParams.times]

    # Itereate over each file and preprocess them
    print("Processing new model files...")
    for file_idx in range(len(all_path_names)):
        print(F"================ {all_file_names[file_idx]} ================================ ")
        # Read file as xarray
        if mode == wrfFileType.old:
            # Verify that variable names contais PH
            if 'PH' in variable_names:
                # Generate a new set of path names replacing c1h to c3h
                all_path_names_3h = [x.replace('c1h', 'c3h') for x in all_path_names]
                cur_xr_ds = xr.open_dataset(all_path_names[file_idx], decode_times=False)
                cur_xr_ds_3h = xr.open_dataset(all_path_names_3h[file_idx], decode_times=False)
                # Read only PH from cur_xr_ds_3h
                cur_xr_ds_PH = cur_xr_ds_3h['PH']
                # Manually interpolate the time dimension
                interp_times = [np.round(x/3,4) for x in range(24)]
                cur_xr_ds_PH = cur_xr_ds_PH.interp(Time=interp_times)
                cur_xr_ds['PH'] = cur_xr_ds_PH
        else:
            cur_xr_ds = xr.open_dataset(all_path_names[file_idx], decode_times=False)

        if 'PH' in variable_names:
            # TODO Hardcoded level 10 selectoin for PH
            # Erika Jimenez dijo que en el pronóstico viejo era entre 32 y 34 (o sea 33), creo questas son del nuevo. Pero mañana te paso los detalles.
            ph_level = 33
            cur_xr_ds['PH'] = cur_xr_ds['PH'].sel(bottom_top_stag=ph_level)
            # Assign the modified variable back to the same variable in the dataset
            # xarray_ds['data_var'] = data_var_at_depth
            # Plot PH
            # import matplotlib.pyplot as plt
            # plt.imshow(cur_xr_ds['PH'].values[0,:,:])
            # plt.savefig('PHNew.png')

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

        # Subsampling the data
        for output_size in output_sizes:
            output_folder_final = F"{output_folder}_{output_size['rows']}_{output_size['cols']}"
            if not (os.path.exists(output_folder_final)):
                os.makedirs(output_folder_final)

            try:
                subsampled_xr_ds, coarselat, coarselon = subsampleData(cropped_xr_ds, variable_names, output_size['rows'], output_size['cols'])
            except Exception as e:
                print(F"ERROR!!!!! Failed to subsample file {all_path_names[file_idx]}, output size: {output_size}: {e}")
                continue

            # For debugging, visualizing results
            # print("\tVisualizing cropped results...")

            # time_to_plot = 23
            # file_text = f"{all_file_names[file_idx].split('_')[3]}_{output_size['rows']}_{output_size['cols']}_{time_to_plot}"
            # field_to_plot = 'T2'
            # viz_obj = EOAImageVisualizer(lats=LAT, lons=LON, disp_images=True, output_folder=output_folder_imgs)
            # viz_obj.plot_3d_data_npdict(cur_xr_ds, var_names=[field_to_plot],
            #                                 z_levels=[time_to_plot], title='Original Data', file_name_prefix=f'Original_{file_text}')
            # viz_obj = EOAImageVisualizer(lats=newLAT, lons=newLon, disp_images=True, output_folder=output_folder_imgs)
            # viz_obj.plot_3d_data_npdict(cropped_xr_ds, var_names=[field_to_plot],
            #                                 z_levels=[time_to_plot], title='Cropped Data', file_name_prefix=f'Cropped_{file_text}')
            # viz_obj = EOAImageVisualizer(lats=coarselat, lons=coarselon, disp_images=True, output_folder=output_folder_imgs)
            # viz_obj.plot_3d_data_npdict(subsampled_xr_ds, var_names=[field_to_plot],
            #                                 z_levels=[time_to_plot], title='Coarsened Data', file_name_prefix=f'Coarsened_{file_text}')

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
    print(F"Processing year {year}")
    user_config = getPreprocWRFParams()

    input_folder = user_config[PreprocParams.input_folder_new]
    input_folder_old = user_config[PreprocParams.input_folder_old]

    start_date = F'{year}-01-01'
    end_date = F'{year + 1}-01-01'

    if year < 2017: # We use the 'old' model
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
    start_year = 2024
    end_year = 2025

    # Run sequential
    runParallel(2024)

    # Run this process in parallel splitting separating by years
    # NUMBER_PROC = 10
    # p = Pool(NUMBER_PROC)
    # p.map(runParallel, range(start_year, end_year))