from conf.params import MergeFilesParams, LocalTrainingParams
from conf.localConstants import constants
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os

from constants.AI_params import ModelParams, AiModels, TrainingParams, ClassificationParams, ClassificationMetrics

output_folder = '/data/UNAM/Air_Pollution_Forecast/Data'
# output_folder = '/home/olmozavala/REMOTE_PROJECTS/OUTPUT'

# =================================== TRAINING ===================================
# ----------------------------- UM -----------------------------------
_run_name = F'Adam_AllStations_300_300_200_100'  # Name of the model, for training and classification

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: True,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.HIDDEN_LAYERS: 4,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [300, 300, 200, 100],
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 34,  # 34 stations
    }
    return {**cur_config, **model_config}


def getTrainingParams():
    cur_config = {
        TrainingParams.input_folder: output_folder,
        TrainingParams.output_folder: F"{join(output_folder, constants.training_output_folder.value)}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.evaluation_metrics: [metrics.mean_squared_error],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: losses.mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 1024,
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False,
        LocalTrainingParams.station: "AJU",
        # LocalTrainingParams.station: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
        #  ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
        #  ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
        #  ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
        #  , "XCH"],
        LocalTrainingParams.pollutant: "cont_otres",
        LocalTrainingParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: 16,  # 4x4
        LocalTrainingParams.num_hours_in_netcdf: 72
    }
    return append_model_params(cur_config)

def getMergeParams():
    # We are using the same parameter as the
    cur_config = {
        MergeFilesParams.input_folder: output_folder,
        MergeFilesParams.output_folder: F"{join(output_folder, constants.merge_output_folder.value)}",
        # MergeFilesParams.stations: ["ACO", "AJM"],
        MergeFilesParams.stations: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
          ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
          ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
          ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
          , "XCH"],
        MergeFilesParams.pollutant_tables: ["cont_otres"],
        MergeFilesParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: 16,
        LocalTrainingParams.num_hours_in_netcdf: 72,
        MergeFilesParams.output_folder: join(output_folder, constants.merge_output_folder.value)
    }

    return cur_config

def get_usemodel_1d_config():
    models_folder = '/data/UNAM/Air_Pollution_Forecast/Data/Training/models'
    # model_file = 'Relu_Sigmoid_2020_02_10_19_35_cont_otres_AllStations-480-0.00447.hdf5'
    model_file = 'Adam_AllStations_300_300_200_100_2020_02_10_20_32_cont_otres_AllStations-553-0.00411.hdf5'
    cur_config = {
        ClassificationParams.input_file: join(output_folder, constants.merge_output_folder.value, 'cont_otres_AllStations.csv'),
        ClassificationParams.output_folder: F"{join(output_folder, 'Results')}",
        ClassificationParams.model_weights_file: join(models_folder, model_file),
        ClassificationParams.output_file_name: 'Results.csv',
        ClassificationParams.output_imgs_folder: F"{join(output_folder, 'Results', _run_name)}",
        ClassificationParams.show_imgs: True,
        ClassificationParams.save_prediction: True,
        LocalTrainingParams.forecasted_hours: 24,
        ClassificationParams.metrics: [ClassificationMetrics.MSE],
        TrainingParams.config_name: _run_name,
    }
    return append_model_params(cur_config)
