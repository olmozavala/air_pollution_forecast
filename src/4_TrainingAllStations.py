from constants.AI_params import TrainingParams, ModelParams
from conf.localConstants import constants
from conf.TrainingUserConfiguration import getTrainingParams
from inout.io_common import create_folder
from conf.params import LocalTrainingParams
from proj_preproc.preproc import normalizeAndFilterData, generate_date_hot_vector
import trainingutils as utilsNN
from models.modelSelector import select_1d_model

from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from os.path import join

tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)

def main():
    config = getTrainingParams()
    # =============== Read data and merge meteorological variables===============
    print("Reading data")
    pollutant = config[LocalTrainingParams.pollutant]

    input_folder = config[TrainingParams.input_folder]
    output_folder = config[TrainingParams.output_folder]

    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    model_name_user = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    years = config[LocalTrainingParams.years]
    debugging = config[LocalTrainingParams.debug]
    filter_stations = config[LocalTrainingParams.stations]
    filter_dates = config[LocalTrainingParams.filter_dates]

    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)


    data = None
    for year in years:
        print(F"============ Reading data for {year}: {pollutant} -- AllStations ==========================")
        if debugging:
            db_file_name = join(input_folder, F"{year}_{pollutant}_AllStationsDebug.csv")
        else:
            db_file_name = join(input_folder, F"{year}_{pollutant}_AllStations.csv")

        temp = pd.read_csv(db_file_name, index_col=0, parse_dates=True)
        if data is None:
            all_data_cols = temp.columns
            date_columns = [x for x in all_data_cols if (x.find('week') != -1) or (x.find('hour') != -1) or (x.find('year') != -1)]
            stations_columns = [x for x in all_data_cols if (x.find('h') == -1) and (x not in date_columns)]
            meteo_columns = [x for x in all_data_cols if (x.find('h') != -1) and (x not in date_columns) and (x not in stations_columns)]
            desired_columns = meteo_columns + filter_stations + date_columns
            data = temp[desired_columns]
        else:
            data = data.append(temp[desired_columns])

    print("Appending date hot vector...")
    date_hv = generate_date_hot_vector(data.index)
    data = pd.concat([data, date_hv], axis=1)
    print("Done!")

    # ********** Restricting only data between the hours of 9 to 20 TODO hardoded *****
    if filter_dates:
        filtered_data = data.between_time("9:00", "20:00")
    else:
        filtered_data = data
    datetimes_str = filtered_data.index.values

    data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns =\
        normalizeAndFilterData(filtered_data, datetimes_str, forecasted_hours, output_folder=parameters_folder,
                               run_name=model_name_user, read_from_file=False)

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

    # ****** Bootstrap everything above 60 ppts TODO hardoded
    idx_by_col = Y_df > 0.24
    idx_above = idx_by_col.any(axis=1)
    # butstrap_size = 5 # How many times are we repeating the 'high' values
    # for i in range(butstrap_size):
    Y = np.append(Y, Y[idx_above, :], axis=0)
    X = np.append(X, X[idx_above, :], axis=0)

    config[ModelParams.INPUT_SIZE] = len(X_df.columns)
    print(F'Data shape: {filtered_data.shape} Data axes {filtered_data.axes}')
    print(F'X shape: {X.shape} Y shape: {Y.shape}')

    tot_examples = X.shape[0]
    rows_to_read = np.arange(tot_examples)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    print("Train examples (total:{}) :{}".format(len(train_ids), rows_to_read[train_ids]))
    print("Validation examples (total:{}) :{}:".format(len(val_ids), rows_to_read[val_ids]))
    print("Test examples (total:{}) :{}".format(len(test_ids), rows_to_read[test_ids]))

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{now}_{pollutant}_AllStations'

    # ******************* Selecting the model **********************
    config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = Y.shape[1]
    print(F"Nomber of output variables {Y.shape[1]}")
    model = select_1d_model(config)
    plot_model(model, to_file=join(output_folder, F'{model_name}.png'), show_shapes=True)

    file_name_splits = join(split_info_folder, F'{model_name}.csv')
    utilsNN.save_splits(file_name_splits, train_ids, val_ids, test_ids)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Training ...")
    # This part should be somehow separated, it will change for every project
    x_train = X[train_ids, :]
    y_train = Y[train_ids, :]
    x_val = X[val_ids, :]
    y_val = Y[val_ids, :]

    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        callbacks=[logger, save_callback, stop_callback])

    # Evaluate all the groups (train, validation, test)
    # Unormalize and plot

if __name__ == '__main__':
    main()
