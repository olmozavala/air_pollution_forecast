from conf.params import MergeFilesParams, LocalTrainingParams
import glob
from conf.localConstants import constants
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.activations as activations
import tensorflow.keras.losses as losses
from os.path import join
import os
from sklearn.metrics import *
from proj_prediction.metrics import restricted_mse
import numpy as np

from ai_common.constants.AI_params import ModelParams, AiModels, TrainingParams, ClassificationParams, VisualizationResultsParams, NormParams

all_stations = ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
          ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
          ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
          ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
          , "XCH"]

stations_2020 = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                 "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                 "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]

data_folder = '/ZION/AirPollutionData/Data/'
# data_folder = '/data/PollutionData/'
training_output_folder = '/ZION/AirPollutionData/Training'
grid_size = 4
merged_specific_folder = f'{grid_size*grid_size}' # We may have multiple folders inside merge depending on the cuadrants
filter_training_hours = False
start_year = 2010
end_year = 2019
_test_year = 2015
_debug = False

# =================================== TRAINING ===================================
# ----------------------------- UM -----------------------------------
_run_name = F'2023'  # Name of the model, for training and classification

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: True,
        ModelParams.BATCH_NORMALIZATION: True,
        # ModelParams.CELLS_PER_HIDDEN_LAYER: [300, 300, 300],
        ModelParams.CELLS_PER_HIDDEN_LAYER: [300, 300, 200, 100, 100, 100, 100, 100, 100],
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 1,
        ModelParams.ACTIVATION_HIDDEN_LAYERS: 'relu',
        ModelParams.ACTIVATION_OUTPUT_LAYERS: None
    }
    model_config[ModelParams.HIDDEN_LAYERS] = len(model_config[ModelParams.CELLS_PER_HIDDEN_LAYER])
    return {**cur_config, **model_config}


def getMergeParams():
    # We are using the same parameter as the
    cur_config = {
        MergeFilesParams.input_folder: data_folder,
        # MergeFilesParams.stations: ["ACO", "AJM"],
        MergeFilesParams.stations: stations_2020,
        MergeFilesParams.pollutant_tables: ["cont_otres"],
        MergeFilesParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: grid_size * grid_size,
        LocalTrainingParams.num_hours_in_netcdf: 24, # 72 (forecast)
        MergeFilesParams.output_folder: join(data_folder, constants.merge_output_folder.value),
        MergeFilesParams.years: range(2010,2023)
    }

    return cur_config


def getTrainingParams():
    cur_config = {
        TrainingParams.input_folder: join(data_folder, constants.merge_output_folder.value, merged_specific_folder),
        TrainingParams.output_folder: F"{join(data_folder, constants.training_output_folder.value)}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: 0, # We will test with a diferent day
        TrainingParams.evaluation_metrics: [metrics.mean_squared_error],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: losses.mean_squared_error,  # Loss function to use for the learning
        # TrainingParams.loss_function: metrics.mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 200,
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False,
        TrainingParams.normalization_type: NormParams.mean_zero,
        LocalTrainingParams.stations: stations_2020,
        LocalTrainingParams.pollutants: ["cont_otres"],
        LocalTrainingParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: 64,  # 8x8
        LocalTrainingParams.num_hours_in_netcdf: 24,
        LocalTrainingParams.years: range(start_year, end_year),
        LocalTrainingParams.debug: _debug,
        LocalTrainingParams.filter_dates: filter_training_hours
    }
    return append_model_params(cur_config)


models_folder = join(training_output_folder, 'models')
splits_folder = join(training_output_folder, 'Splits')

def get_test_file(debug=False):
    year = _test_year
    if debug:
        test_file = join(data_folder, constants.merge_output_folder.value,
                         merged_specific_folder, F'{year}_cont_otres_AllStationsDebug.csv')
    else:
        test_file = join(data_folder, constants.merge_output_folder.value,
                         merged_specific_folder, F'{year}_cont_otres_AllStations.csv')
    return test_file

def get_makeprediction_config():

    results_folder = 'Results'
    cur_config = {
        ClassificationParams.input_file: get_test_file(debug=_debug),
        ClassificationParams.output_folder: F"{join(data_folder, results_folder)}",
        ClassificationParams.model_weights_file: models_folder,  # We are only passing the folder, the model will be loaded automatically
        # ClassificationParams.split_file: join(splits_folder, F"{run_name}.csv"),
        ClassificationParams.split_file: '',
        ClassificationParams.output_file_name: join(training_output_folder,results_folder, F'{_run_name}.csv'),
        ClassificationParams.output_imgs_folder: F"{join(data_folder, results_folder, _run_name)}",
        ClassificationParams.generate_images: False,
        ClassificationParams.show_imgs: False,
        ClassificationParams.save_prediction: True,
        LocalTrainingParams.stations: stations_2020,
        LocalTrainingParams.pollutants: ['otres'],
        LocalTrainingParams.forecasted_hours: 24,
        ClassificationParams.metrics: {'rmse': mean_squared_error,
                                       'mae': mean_absolute_error,
                                       'r2': r2_score,
                                       # 'r': np.corrcoef,
                                       'ex_var': explained_variance_score},
        TrainingParams.config_name: _run_name,
        LocalTrainingParams.filter_dates: filter_training_hours
    }
    return append_model_params(cur_config)

def get_visualization_config():
    file_name = _run_name
    cur_config = {
        VisualizationResultsParams.gt_data_file: get_test_file(debug=_debug),
        VisualizationResultsParams.nn_output: join(training_output_folder, 'Results',
                                                   F'{file_name}_nnprediction.csv'),
        VisualizationResultsParams.nn_metrics: join(training_output_folder, 'Results',
                                                   F'{file_name}.csv'),
        LocalTrainingParams.stations: stations_2020,
        TrainingParams.config_name: _run_name,
    }
    return cur_config

