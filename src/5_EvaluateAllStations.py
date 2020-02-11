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

from conf.TrainingUserConfiguration import get_usemodel_1d_config
from inout.io_common import  create_folder
from data_generation.utilsDataFormat import *
from models.modelSelector import select_1d_model
from metrics import numpy_dice
from os import listdir

def main():
    config = get_usemodel_1d_config()
    # *********** Reads the parameters ***********

    input_file = config[ClassificationParams.input_file]
    output_folder = config[ClassificationParams.output_folder]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]
    output_file_name = config[ClassificationParams.output_file_name]
    run_name = config[TrainingParams.config_name]
    model_weights_file = config[ClassificationParams.model_weights_file]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    disp_images = config[ClassificationParams.show_imgs]

    # ********** Reading and preprocessing data *******
    _all_stations = ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY",
                     "CUA" , "CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA",
                     "LAG", "LLA" , "LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT",
                     "SAG", "SFE" , "SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL",
                     "VIF", "XAL" , "XCH"]

    # Iterate over the stations
    # Selects the proper model file for the current station
    assert len(model_weights_file) > 0
    assert len(input_file) > 0

    print(F"Working with: {model_weights_file} and {input_file}")

    data = pd.read_csv(input_file, index_col=0)

    datetimes_str = data.index.values
    datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

    stations_columns = [x for x in data.columns.values if x.find('h') == -1]
    meteo_columns = [x for x in data.columns.values if x.find('h') != -1]

    config[ModelParams.INPUT_SIZE] = len(data.columns)
    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")

    # Predicting for the next value after 24hrs (only one)
    print("Normalizing data....")
    # Normalizing meteorological variables
    min_values_meteo = data[meteo_columns].min()
    max_values_meteo = data[meteo_columns].max()
    # Manually setting the min/max values for the pollutant (ozone)
    min_values_pol = 0
    max_values_pol = 150

    data_norm_df = data.copy()
    data_norm_df[meteo_columns] = (data_norm_df[meteo_columns] - min_values_meteo) / (
            max_values_meteo - min_values_meteo)
    data_norm_df[stations_columns] = (data_norm_df[stations_columns] - min_values_pol) / (
            max_values_pol - min_values_pol)
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

    # Replacing nan columns with the mean value of all the other columns
    mean_values = data_norm_df[stations_columns].mean(1)

    data_norm_df_final = data_norm_df.copy()
    for cur_station in stations_columns:
        data_norm_df_final[cur_station] = data_norm_df[cur_station].fillna(mean_values)

    X = data_norm_df_final.loc[datetimes_str[accepted_times_idx]]
    Y = data_norm_df_final.loc[datetimes_str[y_times_idx]][stations_columns]
    print(F'X shape: {X.shape} Y shape: {Y.shape}')

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_1d_model(config)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    create_folder(output_folder)
    create_folder(output_imgs_folder)

    # *********** Makes a dataframe to contain the DSC information **********
    metrics_params = config[ClassificationParams.metrics]
    metrics_dict = {met.name: met.value for met in metrics_params}

    # *********** Iterates over each case *********
    t0 = time.time()
    # -------------------- Reading data -------------
    output_nn_all = model.predict(X, verbose=1)

    # Plotting some intermediate results
    import matplotlib.pyplot as plt
    plot_this_many= 24*60
    # start = np.random.randint(0, len(data) - plot_this_many)
    start = 0
    end = start + plot_this_many
    plt.figure(figsize=[64, 8])
    viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=disp_images)
    X_plot = range(plot_this_many)
    for id_station, cur_station in enumerate(stations_columns):
        Y_plot = Y[cur_station].values
        viz_obj.plot_1d_data_np(X_plot, [Y_plot[start:end], output_nn_all[start:end,id_station]],
                                title=F"{cur_station} ozone",
                                labels=['Original', 'Forecasted'],
                                wide_ratio=4,
                                file_name_prefix=F"{run_name}_{cur_station}_all")

    print(F'\t Done! Elapsed time {time.time() - t0:0.2f} seg')

if __name__ == '__main__':
    main()
