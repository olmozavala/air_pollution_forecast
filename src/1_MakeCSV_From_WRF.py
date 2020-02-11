from conf.MakeWRF_CSV_UserConfiguration import getPreprocWRFParams
from conf.params import PreprocParams
from proj_preproc.wrf import crop_variables_xr, subsampleData
from proj_preproc.utils import getStringDates
from conf.localConstants import constants
import os
import xarray as xr
# from img_viz.eoa_viz import EOAImageVisualizer

from io_netcdf.inout import read_wrf_files_names, saveFlattenedVariables

def main():

    # Reads user configuration
    user_config = getPreprocWRFParams()
    variable_names = user_config[PreprocParams.variables]
    input_folder = user_config[PreprocParams.input_folder]
    output_folder = user_config[PreprocParams.output_folder]
    output_folder_imgs= user_config[PreprocParams.output_imgs_folder]
    output_sizes = user_config[PreprocParams.resampled_output_sizes]
    bbox = user_config[PreprocParams.bbox]
    times = user_config[PreprocParams.times]
    start_date = user_config[PreprocParams.start_date]
    end_date = user_config[PreprocParams.end_date]

                    # viz_obj = EOAImageVisualizer(output_folder=output_folder_imgs, disp_images=True)

    # Reads all netetCDF files
    print("Reading file names...")
    all_dates, all_file_names , all_path_names = read_wrf_files_names(input_folder, start_date, end_date)
    print("Done!")

    # Itereate over each file and preprocess them
    print("Processing files...")
    for file_idx in range(len(all_path_names)):
        print(F"================ {all_file_names[file_idx]} ================================ ")
        # Read file as xarray
        cur_xr_ds = xr.open_dataset(all_path_names[file_idx])
        # Printing the summary of the data
        # viz_obj.xr_summary(cur_xr_ds)
        print(F"\tCropping...")
        # Crops the desired variable_names
        try:
            cropped_xr_ds, newLAT, newLon = crop_variables_xr(cur_xr_ds, variable_names, bbox, times=times)
        except Exception as e:
            print(F"ERROR!!!!! Failed to crop file {all_path_names[file_idx]}: {e}")
            continue

        # viz_obj.xr_summary(cropped_xr_ds)
        print("\tDone!")

        for output_size in output_sizes:
            output_folder_final = F"{output_folder}_{output_size['rows']}_{output_size['cols']}"
            if not (os.path.exists(output_folder_final)):
                os.makedirs(output_folder_final)
            # Subsample the data
            print(F"\tSubsampling...")

            try:
                subsampled_xr_ds = subsampleData(cropped_xr_ds, variable_names, output_size['rows'], output_size['cols'])
            except Exception as e:
                print(F"ERROR!!!!! Failed to subsample file {all_path_names[file_idx]}, output size: {output_size}: {e}")
                continue
            # viz_obj.xr_summary(subsampled_xr_ds)
            print("\tDone!")

            print("\tVisualizing results...")
            # For debugging, visualizing results
            # viz_obj.plot_3d_data_xarray_map(cur_xr_ds, var_names=[variable_names[0]],
            #                                 timesteps=[0], title='Original Data', file_name_prefix='Original')
            # viz_obj.plot_3d_data_xarray_map(cropped_xr_ds, var_names=variable_names,
            #                                 timesteps=[0,1], title='Cropped Data', file_name_prefix='Cropped')
            # viz_obj.plot_3d_data_xarray_map(subsampled_xr_ds, var_names=variable_names,
            #                                 timesteps=[0,1], title='Subsampled Data', file_name_prefix='Subsampled')

            print("\tFlattening variables and saving as csv")
            # Obtain time strings for current file
            # Save variables as a single CSV file

            try:
                saveFlattenedVariables(subsampled_xr_ds, variable_names, output_folder_final,
                                       file_name=F"{all_dates[file_idx].strftime(constants.date_format.value)}.csv",
                                       index_names=getStringDates(all_dates[file_idx], times),
                                       index_label=constants.index_label.value)
            except Exception as e:
                print(F"ERROR!!!!! Failed to save file {all_path_names[file_idx]}: {e}")
            continue

if __name__== '__main__':
    main()
