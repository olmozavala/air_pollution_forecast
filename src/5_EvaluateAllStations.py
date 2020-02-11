from datetime import datetime, timedelta

from img_viz.eoa_viz import EOAImageVisualizer
from sklearn import preprocessing
from conf.localConstants import constants
import os
from pandas import DataFrame
import pandas as pd
import time
from constants.AI_params import ModelParams, ClassificationParams, TrainingParams
from os.path import join
from conf.params import LocalTrainingParams

from conf.TrainingUserConfiguration import get_makeprediction_config
from inout.io_common import  create_folder
from data_generation.utilsDataFormat import *
from models.modelSelector import select_1d_model
from metrics import numpy_dice
from os import listdir
from proj_preproc.preproc import normalizeAndFilterData, deNormalize
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
    metrics_user = config[ClassificationParams.metrics]


    # Iterate over the stations
    # Selects the proper model file for the current station
    assert len(model_weights_file) > 0
    assert len(input_file) > 0

    print(F"Working with: {model_weights_file} and {input_file}")

    data = pd.read_csv(input_file, index_col=0)
    datetimes_str = data.index.values

    config[ModelParams.INPUT_SIZE] = len(data.columns)
    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")

    print(F'Normalizing and filtering data....')
    data_norm_df_final, accepted_times_idx, y_times_idx, stations_columns, meteo_columns = \
        normalizeAndFilterData(data, datetimes_str, forecasted_hours)

    X = data_norm_df_final.loc[datetimes_str[accepted_times_idx]]
    Y = data_norm_df_final.loc[datetimes_str[y_times_idx]][stations_columns]
    print(F'X shape: {X.shape} Y shape: {Y.shape}')

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_1d_model(config)

    # *********** Chooses the proper model ***********
    print('Reading splits info....')
    split_info = pd.read_csv(splits_file, dtype=np.int16)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    create_folder(output_folder)
    create_folder(output_imgs_folder)

    # ************ Makes NN Prediction ********
    print('Making prediction ....')
    output_nn_all = model.predict(X, verbose=1)

    # ************ Recovering original units********
    print('Recovering original units....')
    nn_df = pd.DataFrame(output_nn_all, columns=stations_columns, index=data.index[y_times_idx])
    nn_original_units = deNormalize(nn_df)
    Y_original = deNormalize(Y)

    # ************ Computing metrics********
    print('Computing metrics....')
    metrics_result = compute_metrics(Y_original, nn_original_units, metrics_user, split_info, output_file_name, stations_columns)

if __name__ == '__main__':
    main()
