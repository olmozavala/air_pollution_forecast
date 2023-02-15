from datetime import date, datetime, timedelta

from viz_utils.eoa_viz import EOAImageVisualizer
from sklearn import preprocessing
from conf.localConstants import constants
from pandas import DataFrame
import pandas as pd
import time
from ai_common.constants.AI_params import *
from os.path import join
from conf.params import LocalTrainingParams

from conf.TrainingUserConfiguration import get_makeprediction_config
from io_utils.io_common import  create_folder
from AI.data_generation.utilsDataFormat import *
from ai_common.models.modelSelector import select_1d_model
from os import listdir
import glob
import os

config = get_makeprediction_config()
# *********** Reads the parameters ***********

input_file = config[ClassificationParams.input_file]
output_folder = config[ClassificationParams.output_folder]
output_imgs_folder = config[ClassificationParams.output_imgs_folder]
output_file_name = config[ClassificationParams.output_file_name]
model_weights_file = config[ClassificationParams.model_weights_file]
forecasted_hours = config[LocalTrainingParams.forecasted_hours]
pollutant = config[LocalTrainingParams.pollutants][0]

# ********** Reading and preprocessing data *******
all_stations = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                 "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                 "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]

evaluate_stations = ["UIZ", "AJU", "ATI"]

# Iterate over the stations
models_folder = '/data/PollutionData/Training/models/'
data_folder = '/data/PollutionData/MergedDataCSV/8_8/'

test_year = 2019

for cur_station in evaluate_stations:
    # try:
    model_files = glob.glob(join(models_folder, F'*{cur_station}*'))
    model_files.sort(key=os.path.getmtime)
    model_weights_file = model_files[-1]
    input_file = join(data_folder,f"{test_year}_cont_otres_AllStations.csv")

    # Selects the proper model file for the current station
    assert len(model_weights_file) > 0
    assert len(input_file) > 0

    print(F"Working with: {model_weights_file} and input: {input_file}")

    data = pd.read_csv(input_file, index_col=0)
    # -------- Removing not used stations
    remove_columns = [x for x in all_stations if x.find(cur_station) == -1]
    data = data.drop(columns=remove_columns)
    # -------- Remove nans

    config[ModelParams.INPUT_SIZE] = len(data.columns)
    print(F'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")

    # Predicting for the next value after 24hrs (only one)
    print("Normalizing data....")
    # TODO we need to normalize by reading the parameters from the training
    # norm_file = "/data/PollutionData/Training/Parameters/2023_cont_otres_UIZ_2023_02_14_21_19.csv"
    datetimes_str = data.index.values
    datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

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
    Y_df = data_norm_df.loc[datetimes_str[y_times_idx]][cur_station]
    X = X_df.values
    Y = Y_df.values

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
    metrics_dict = config[ClassificationParams.metrics]

    # *********** Iterates over each case *********
    t0 = time.time()
    # -------------------- Reading data -------------
    output_nn_all = model.predict(X, verbose=1)

    # Plotting some intermediate results
    import matplotlib.pyplot as plt
    size = 24 * 10  # Two months of data
    start = np.random.randint(0, len(data) - size)
    end = start + size
    plt.figure(figsize=[64, 8])
    x_plot = range(len(Y))
    y_plot = Y
    yy_plot = Y_df.iloc[start:end].values
    plot_this_many= 24*60
    plt.plot(x_plot[0:plot_this_many], y_plot[0:plot_this_many], label='Original')
    plt.plot(x_plot[0:plot_this_many], output_nn_all[0:plot_this_many,0], label='Forecasted')
    plt.show()

    print(F'\t Done! Elapsed time {time.time() - t0:0.2f} seg')

    # except Exception as e:
    #     print(F"---------------------------- Failed {cur_station} error: {e} ----------------")


##

