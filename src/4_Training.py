from img_viz.eoa_viz import EOAImageVisualizer

from conf.localConstants import constants

from conf.TrainingUserConfiguration import getTrainingParams
from conf.params import LocalTrainingParams
from preproc.constants import NormParams

from training.utils import getQuadrantsAsString
from AI.data_generation.Generators3D import *
from datetime import date, datetime, timedelta
import os
import matplotlib.pyplot as plt

from inout.io_common import create_folder

import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

from conf.AI_params import *
import AI.trainingutils as utilsNN
import AI.models.modelBuilder3D as model_builder
from AI.models.modelSelector import select_1d_model

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def main():
    config = getTrainingParams()
    # =============== Read data and merge meteorological variables===============
    print("Reading data")
    station = config[LocalTrainingParams.station]
    pollutant = config[LocalTrainingParams.pollutant]
    # Iterate over all pollutants
    # for cur_pollutant in pollutants:
        # Iterate over all stations
        # for cur_station in stations:
    try:
        trainModel(config, pollutant, station)
    except Exception as e:
        print(F"ERROR! It has failed for:{pollutant} -- {station}: {e}")

def trainModel(config, cur_pollutant, cur_station):
    """Trying to separate things so that tf 'cleans' the memory """

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
    data_augmentation = config[TrainingParams.data_augmentation]
    start_date = config[LocalTrainingParams.start_date]
    end_date = config[LocalTrainingParams.end_date]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    num_quadrants = config[LocalTrainingParams.tot_num_quadrants]
    num_hours_in_netcdf  = config[LocalTrainingParams.num_hours_in_netcdf]

    model_type = config[ModelParams.MODEL]

    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    imgs_folder= join(output_folder, 'imgs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    WRF_data_folder_name = join(input_folder,constants.wrf_output_folder.value,
                                F"{constants.wrf_each_quadrant_name.value}_{getQuadrantsAsString(num_quadrants)}")

    append_meteo_colums = True  # It is used to append the meteo variable columns into the database

    viz_obj = EOAImageVisualizer(output_folder=imgs_folder, disp_images=False)

    print(F"============ Reading data for: {cur_pollutant} -- {cur_station} ==========================")
    db_file_name = join(input_folder, constants.merge_output_folder.value, F"{cur_pollutant}_{cur_station}.csv")
    data = pd.read_csv(db_file_name, index_col=0)

    config[ModelParams.INPUT_SIZE] = len(data.columns)
    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")

    # Predicting for the next value after 24hrs (only one)
    print("Normalizing data....")
    datetimes_str = data.index.values
    datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])
    dates = np.array([cur_datetime.date() for cur_datetime in datetimes])

    scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(data)
    data_norm_np = scaler.transform(data)
    data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)
    print(F'Done!')

    # Filtering only dates where there is data "forecasted hours after" (24 hrs after)
    print(F"\tBuilding X and Y ....")
    accepted_times_idx = []
    y_times_idx = []
    for i, c_datetime in enumerate(datetimes):
        forecasted_datetime = (c_datetime + timedelta(hours=forecasted_hours))
        if forecasted_datetime in datetimes:
            accepted_times_idx.append(i)
            y_times_idx.append(np.argwhere(forecasted_datetime == datetimes)[0][0])

    X_df = data_norm_df.loc[datetimes_str[accepted_times_idx]]
    Y_df = data_norm_df.loc[datetimes_str[y_times_idx]][cur_pollutant]
    X = X_df.values
    Y = Y_df.values

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
    model_name = F'{model_name_user}_{now}_{cur_pollutant}_{cur_station}'

    # ******************* Selecting the model **********************
    model = select_1d_model(config)
    plot_model(model, to_file=join(output_folder, F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, folders_to_read=rows_to_read,
                        train_idx=train_ids, val_idx=val_ids, test_idx=test_ids)

    print(F"Norm params: {scaler.get_params()}")
    file_name_normparams = join(parameters_folder, F'{model_name}.txt')
    utilsNN.save_norm_params(file_name_normparams, NormParams.min_max, scaler)

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

    # Plotting some intermediate results
    import matplotlib.pyplot as plt
    size = 24 * 60  # Two months of data
    start = np.random.randint(0, len(data) - size)
    end = start + size
    plt.figure(figsize=[64, 8])
    x_plot = range(len(X_df.iloc[start:end].index.values))
    y_plot = X_df.iloc[start:end][cur_pollutant].values
    yy_plot = Y_df.iloc[start:end].values
    viz_obj.plot_1d_data_np(x_plot, [y_plot, yy_plot], title=F"{cur_pollutant}_{cur_station}",
                            labels=['Current', 'Desired'],
                            file_name_prefix=F"{cur_pollutant}_{cur_station}")

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
