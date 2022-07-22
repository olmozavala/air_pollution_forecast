from datetime import date, datetime, timedelta

from img_viz.eoa_viz import EOAImageVisualizer
from sklearn import preprocessing
from conf.localConstants import constants
import os
from pandas import DataFrame
import pandas as pd
import time
from conf.AI_params import *
from os.path import join
from conf.params import LocalTrainingParams

from img_viz.medical import MedicalImageVisualizer
from conf.TrainingUserConfiguration import get_makeprediction_config
from inout.io_common import  create_folder
from AI.data_generation.utilsDataFormat import *
from AI.models.modelSelector import select_1d_model
from AI.metrics import numpy_dice
from os import listdir

def main():
    config = get_makeprediction_config()
    # *********** Reads the parameters ***********

    input_file = config[ClassificationParams.input_file]
    output_folder = config[ClassificationParams.output_folder]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]
    output_file_name = config[ClassificationParams.output_file_name]
    model_weights_file = config[ClassificationParams.model_weights_file]
    forecasted_hours = config[LocalTrainingParams.forecasted_hours]
    pollutant = config[LocalTrainingParams.pollutant]

    # ********** Reading and preprocessing data *******
    _all_stations = ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY",
                     "CUA" , "CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA",
                     "LAG", "LLA" , "LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT",
                     "SAG", "SFE" , "SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL",
                     "VIF", "XAL" , "XCH"]

    # Iterate over the stations
    models_folder = '/data/UNAM/Air_Pollution_Forecast/Data/Training/models'
    data_folder = '/data/UNAM/Air_Pollution_Forecast/Data/MergedDataCSV'
    for c_station in _all_stations:
        try:
            model_weights_file = [join(models_folder, x) for x in listdir(models_folder) if x.find(c_station) != -1]
            input_file = [join(data_folder, x) for x in listdir(data_folder) if x.find(c_station) != -1]
            # Selects the proper model file for the current station
            assert len(model_weights_file) > 0
            assert len(input_file) > 0

            print(F"Working with: {model_weights_file} and {input_file}")
            model_weights_file = model_weights_file[0]
            input_file = input_file[0]

            data = pd.read_csv(input_file, index_col=0)

            config[ModelParams.INPUT_SIZE] = len(data.columns)
            print(F'Data shape: {data.shape} Data axes {data.axes}')
            print("Done!")

            # Predicting for the next value after 24hrs (only one)
            print("Normalizing data....")
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
            Y_df = data_norm_df.loc[datetimes_str[y_times_idx]][pollutant]
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
            metrics_params = config[ClassificationParams.metrics]
            metrics_dict = {met.name: met.value for met in metrics_params}

            # *********** Iterates over each case *********
            t0 = time.time()
            # -------------------- Reading data -------------
            output_nn_all = model.predict(X, verbose=1)

            # Plotting some intermediate results
            import matplotlib.pyplot as plt
            size = 24 * 60  # Two months of data
            start = np.random.randint(0, len(data) - size)
            end = start + size
            plt.figure(figsize=[64, 8])
            x_plot = range(len(Y))
            y_plot = Y
            yy_plot = Y_df.iloc[start:end].values
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False)
            plot_this_many= 24*60
            viz_obj.plot_1d_data_np(x_plot[0:plot_this_many], [y_plot[0:plot_this_many], output_nn_all[0:plot_this_many,0]], title=F"{c_station} {pollutant}",
                                    labels=['Original', 'Forecasted'],
                                    wide_ratio=4,
                                    file_name_prefix=F"{pollutant}_{c_station}")

            print(F'\t Done! Elapsed time {time.time() - t0:0.2f} seg')

        except Exception as e:
            print(F"---------------------------- Failed {c_station} error: {e} ----------------")

if __name__ == '__main__':
    main()
