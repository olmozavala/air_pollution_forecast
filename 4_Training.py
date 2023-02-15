from ai_common.constants.AI_params import NormParams, TrainingParams, ModelParams
import ai_common.training.trainingutils as utilsNN
from ai_common.models.modelSelector import select_1d_model

from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_common import create_folder
from conf.localConstants import constants
from conf.TrainingUserConfiguration import getTrainingParams
from conf.params import LocalTrainingParams

from datetime import date, datetime, timedelta
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from os.path import join
import matplotlib.pyplot as plt

tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)


def readData(config, cur_pollutant, start_year, end_year):
    '''
    Reads all the years for the selected pollutant for all the stations
    :param config:
    :param cur_pollutant:
    :param start_year:
    :param end_year:
    :param all_stations:
    :return:
    '''
    input_folder = config[TrainingParams.input_folder]
    # -------- Reading all the years in a single data frame (all stations)
    for c_year in range(start_year, end_year+1):
        db_file_name = join(input_folder, F"{c_year}_{cur_pollutant}_AllStations.csv") # Just for testing
        print(F"============ Reading data for: {cur_pollutant}: {db_file_name}")
        if c_year == start_year:
            data = pd.read_csv(db_file_name, index_col=0)
        else:
            data = pd.concat([data, pd.read_csv(db_file_name, index_col=0)])

    return data

def trainModel(config, cur_pollutant, cur_station, data, all_stations):
    """Trying to separate things so that tf 'cleans' the memory """

    input_folder = config[TrainingParams.input_folder]
    output_folder = config[TrainingParams.output_folder]
    #
    output_folder = join(output_folder,F"{cur_pollutant}_{cur_station}")

    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    model_name_user = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    norm_type = config[TrainingParams.normalization_type]

    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    imgs_folder= join(output_folder, 'imgs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    viz_obj = EOAImageVisualizer(output_folder=imgs_folder, disp_images=False)

    # -------- Removing not used stations
    remove_columns = [x for x in all_stations if x.find(cur_station) == -1]
    data = data.drop(columns=remove_columns)
    # -------- Remove nans
    data = data.dropna()

    config[ModelParams.INPUT_SIZE] = len(data.columns)
    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")

    print("Normalizing data....")
    datetimes_str = data.index.values
    datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

    if norm_type == NormParams.min_max:
        scaler = preprocessing.MinMaxScaler()
    if norm_type == NormParams.mean_zero:
        scaler = preprocessing.StandardScaler()

    scaler = scaler.fit(data)
    data_norm_np = scaler.transform(data)
    data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)
    print(F'Done!')

    # Filtering only dates where there is data "forecasted hours after" (24 hrs after)
    print(F"Building X and Y ....")
    accepted_times_idx = []
    y_times_idx = []
    for i, c_datetime in enumerate(datetimes):
        forecasted_datetime = (c_datetime + timedelta(hours=forecasted_hours))
        if forecasted_datetime in datetimes:
            accepted_times_idx.append(i)
            y_times_idx.append(np.argwhere(forecasted_datetime == datetimes)[0][0])

    X_df = data_norm_df.loc[datetimes_str[accepted_times_idx]]
    Y_df = data_norm_df.loc[datetimes_str[y_times_idx]][cur_station]
    X = X_df.values
    Y = Y_df.values

    print(F'X shape: {X.shape} Y shape: {Y.shape}')

    tot_examples = X.shape[0]
    rows_to_read = np.arange(tot_examples)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc,
                                                                             shuffle_ids=False)

    print("Train examples (total:{}) :{}".format(len(train_ids), rows_to_read[train_ids]))
    print("Validation examples (total:{}) :{}:".format(len(val_ids), rows_to_read[val_ids]))
    print("Test examples (total:{}) :{}".format(len(test_ids), rows_to_read[test_ids]))

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{cur_pollutant}_{cur_station}_{now}'

    # ******************* Selecting the model **********************
    model = select_1d_model(config)
    plot_model(model, to_file=join(output_folder, F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")

    file_name_splits = join(split_info_folder, F'{model_name}.csv')
    info_splits = DataFrame({F'Train({len(train_ids)})': train_ids})
    info_splits[F'Validation({len(val_ids)})'] = 0
    info_splits[F'Validation({len(val_ids)})'][0:len(val_ids)] = val_ids
    info_splits[F'Test({len(test_ids)})'] = 0
    info_splits[F'Test({len(test_ids)})'][0:len(test_ids)] = test_ids
    info_splits.to_csv(file_name_splits, index=None)

    print(F"Norm params: {scaler.get_params()}")
    file_name_normparams = join(parameters_folder, F'{model_name}.csv')
    utilsNN.save_norm_params(file_name_normparams, norm_type, scaler)
    info_splits.to_csv(file_name_splits, index=None)

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
    y_train = Y[train_ids]
    x_val = X[val_ids, :]
    y_val = Y[val_ids]
    x_test = X[test_ids, :]
    y_test = Y[test_ids]

    # ------------------- Plotting some intermediate results
    size = 24 * 10 # 10 days of data
    start = np.random.randint(0, len(data) - size)
    end = start + size
    plt.figure(figsize=[64, 8])
    x_plot = range(len(X_df.iloc[start:end].index.values))
    y_plot = X_df.iloc[start:end][cur_station].values
    yy_plot = Y_df.iloc[start:end].values

    fig, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(x_plot, y_plot, color='r', label='Current')
    ax.plot(x_plot, yy_plot, color='b',  label='Desired')
    ax.set_title = F"{cur_pollutant}_{cur_station}"
    plt.legend()
    plt.show()
    # ------------------- Done Plotting some intermediate results

    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        callbacks=[logger, save_callback, stop_callback])

if __name__ == '__main__':
    config = getTrainingParams()
    stations = config[LocalTrainingParams.stations]
    pollutants = config[LocalTrainingParams.pollutants]
    start_year = 2010
    end_year = 2019
    # It is generating one network for each pollutant for each station
    # Iterate over all pollutants
    for cur_pollutant in pollutants:
        # Read the data for all stations for current pollutant
        data = readData(config, cur_pollutant, start_year, end_year)
        # Iterate over all stations
        for cur_station in stations:
            try:
                trainModel(config, cur_pollutant, cur_station, data, stations)
            except Exception as e:
                print(F"ERROR! It has failed for:{cur_pollutant} -- {cur_station}: {e}")