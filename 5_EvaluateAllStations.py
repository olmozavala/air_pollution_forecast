import sys
sys.path.append("ai_common/")
sys.path.append("eoas_pyutils/")

from datetime import datetime, timedelta

from viz_utils.eoa_viz import EOAImageVisualizer
from conf.localConstants import constants
import numpy as np
from os.path import join, dirname
import pandas as pd
from constants.AI_params import ModelParams, ClassificationParams, TrainingParams
from conf.params import LocalTrainingParams

from conf.TrainingUserConfiguration import get_makeprediction_config
from io_utils.io_common import create_folder
from data_generation.utilsDataFormat import *
from models.modelSelector import select_1d_model
from proj_preproc.preproc import normalizeAndFilterData, deNormalize, generate_date_hot_vector
from proj_prediction.prediction import compute_metrics


def main():
    config = get_makeprediction_config()
    # *********** Reads the parameters ***********

    input_file = config[ClassificationParams.input_file]
    splits_file = config[ClassificationParams.split_file]
    output_folder = config[ClassificationParams.output_folder]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]
    output_file_name = config[ClassificationParams.output_file_name]
    run_name = config[TrainingParams.config_name]
    model_weights_file = config[ClassificationParams.model_weights_file]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    disp_images = config[ClassificationParams.show_imgs]
    generate_images = config[ClassificationParams.generate_images]
    metrics_user = config[ClassificationParams.metrics]
    filter_stations = config[LocalTrainingParams.stations]
    filter_by_dates = config[LocalTrainingParams.filter_dates]

    # Iterate over the stations
    # Selects the proper model file for the current station
    assert len(model_weights_file) > 0
    assert len(input_file) > 0

    print(F"Working with: {model_weights_file} \n and \n {input_file}")

    data = pd.read_csv(input_file, index_col=0, parse_dates=True)

    all_data_cols = data.columns
    date_columns = [x for x in all_data_cols if (x.find('week') != -1) or (x.find('hour') != -1) or (x.find('year') != -1)]
    stations_columns = [x for x in all_data_cols if (x.find('h') == -1) and (x not in date_columns)]
    meteo_columns = [x for x in all_data_cols if (x.find('h') != -1) and (x not in date_columns) and (x not in stations_columns)]
    desired_columns = meteo_columns + filter_stations + date_columns

    print("Appending date hot vector...")
    date_hv = generate_date_hot_vector(data.index)
    data = pd.concat([data[desired_columns], date_hv], axis=1)
    print("Done!")

    if filter_by_dates:
        print("Filtering data to hours 9 to 20...")
        filtered_data =data.between_time("9:00", "20:00")
        # print("Done!")
    else:
        filtered_data = data

    datetimes_str = filtered_data.index.values

    print(F'Normalizing and filtering data....')
    parameters_folder = join(dirname(output_folder), 'Training','Parameters')
    data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns = \
        normalizeAndFilterData(filtered_data, datetimes_str, forecasted_hours, output_folder=parameters_folder,
                               run_name=run_name, read_from_file=True)

    # ********* Filling nan values in the stations with the mean values of all the 'available' stations ********
    X_df = data_norm_df_final.loc[datetimes_str[accepted_times_idx]]
    Y_df = data_norm_df_final.loc[datetimes_str[y_times_idx]][stations_columns]

    # ********* Filling nan values in the stations with the mean values of all the 'available' stations ********
    # for cur_station in stations_columns:
    #     X_df[cur_station] = X_df[cur_station].fillna(X_df['MEAN'])
    #     Y_df[cur_station] = Y_df[cur_station].fillna(data_norm_df_final.loc[datetimes_str[y_times_idx]]['MEAN'])

    # X = data_norm_df_final.loc[datetimes_str[accepted_times_idx]].values
    # X_df = X_df.drop(columns=['MEAN'])
    X_df = X_df.drop(columns=stations_columns)
    X = X_df.values
    # Y = data_norm_df_final.loc[datetimes_str[y_times_idx]][stations_columns].values
    Y = Y_df.values

    config[ModelParams.INPUT_SIZE] = len(X_df.columns)
    print(F'X shape: {X.shape} Y shape: {Y.shape}')

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = Y.shape[1]
    model = select_1d_model(config)

    # *********** Chooses the proper model ***********
    print('Reading splits info....')
    if splits_file != '':  # In this case we do read the information
        split_info = pd.read_csv(splits_file, dtype=np.int16)
    else:
        split_info = pd.DataFrame({'train_ids': [],
                                 'validation_ids':[],
                                 'test_id':[]})
        split_info['train_ids'] = range(Y.shape[0])


    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)


    # ************ Makes NN Prediction ********
    print('Making prediction ....')
    output_nn_all = model.predict(X, verbose=1)

    # ************ Saves raw results ********
    number_of_examples = 10
    if generate_images:
        img_viz = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=disp_images)

        Y[Y == -1] = np.nan # So that we do not show the -1
        for c_example in range(number_of_examples):
            hours_to_plot = 24*3 # How many points to plot
            start_idx = np.random.randint(0, X.shape[0] - hours_to_plot - forecasted_hours)
            end_idx = start_idx + hours_to_plot
            create_folder(output_folder)
            create_folder(output_imgs_folder)
            for idx_station, cur_station in enumerate(filter_stations):
                img_viz.plot_1d_data_np(datetimes_str[y_times_idx][start_idx:end_idx],
                                        [Y[start_idx:end_idx,idx_station],
                                         output_nn_all[start_idx:end_idx, idx_station]],
                                       title=F'{cur_station}', labels=['GT', 'NN'], file_name_prefix=F'{cur_station}_{c_example}')

    # ************ Recovering original units********
    print('Recovering original units....')
    nn_df = pd.DataFrame(output_nn_all, columns=stations_columns, index=filtered_data.index[y_times_idx])
    nn_original_units = deNormalize(nn_df)
    Y_original = deNormalize(Y_df)

    # ************ Computing metrics********
    print('Computing metrics and saving predictions....')
    compute_metrics(Y_original, nn_original_units, metrics_user, split_info, output_file_name, stations_columns)


if __name__ == '__main__':
    main()
