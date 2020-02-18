from conf.params import MergeFilesParams, LocalTrainingParams
from conf.localConstants import constants
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
from sklearn.metrics import *

from constants.AI_params import ModelParams, AiModels, TrainingParams, ClassificationParams, ClassificationMetrics

output_folder = '/data/UNAM/Air_Pollution_Forecast/Data'
training_output_folder = '/data/UNAM/Air_Pollution_Forecast/Data/Training/'
merged_specific_folder = 'Current'
# output_folder = '/home/olmozavala/REMOTE_PROJECTS/OUTPUT'

# =================================== TRAINING ===================================
# ----------------------------- UM -----------------------------------
_run_name = F'2010_2018_Adam_AllStations_300_300_200_100_100_100_100_100_100_TimeHV'  # Name of the model, for training and classification

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: True,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [300, 300, 200, 100, 100, 100, 100, 100, 100],
    }
    model_config[ModelParams.HIDDEN_LAYERS] = len(model_config[ModelParams.CELLS_PER_HIDDEN_LAYER])
    return {**cur_config, **model_config}


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
        LocalTrainingParams.tot_num_quadrants: 64,
        LocalTrainingParams.num_hours_in_netcdf: 72,
        MergeFilesParams.output_folder: join(output_folder, constants.merge_output_folder.value),
        MergeFilesParams.years: range(2018,2020)
    }

    return cur_config


def getTrainingParams():
    cur_config = {
        TrainingParams.input_folder: join(output_folder, constants.merge_output_folder.value, merged_specific_folder),
        TrainingParams.output_folder: F"{join(output_folder, constants.training_output_folder.value)}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: 0, # We will test with a diferent day
        TrainingParams.evaluation_metrics: [metrics.mean_squared_error],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: losses.mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 100,
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False,
        # LocalTrainingParams.station: "AJU",
        # LocalTrainingParams.station: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
        #  ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
        #  ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
        #  ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
        #  , "XCH"],
        LocalTrainingParams.pollutant: "cont_otres",
        LocalTrainingParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: 64,  # 4x4
        LocalTrainingParams.num_hours_in_netcdf: 24
    }
    return append_model_params(cur_config)


def get_makeprediction_config():

    results_folder = 'Results'
    models_folder = join(training_output_folder, 'models')
    splits_folder = join(training_output_folder, 'Splits')
    # model_file = 'Relu_Sigmoid_2020_02_10_19_35_cont_otres_AllStations-480-0.00447.hdf5'
    run_name  = '2010_2018_Adam_AllStations_300_300_200_100_100_100_100_100_100_TimeHV_2020_02_18_18_45_cont_otres_AllStations'
    model_file = F'{run_name}-93-0.00203.hdf5'
    cur_config = {
        ClassificationParams.input_file: join(output_folder, constants.merge_output_folder.value, merged_specific_folder,
                                              '2018_cont_otres_AllStations.csv'),
        ClassificationParams.output_folder: F"{join(output_folder, results_folder)}",
        ClassificationParams.model_weights_file: join(models_folder, model_file),
        # ClassificationParams.split_file: join(splits_folder, F"{run_name}.csv"),
        ClassificationParams.split_file: '',
        ClassificationParams.output_file_name: join(training_output_folder,results_folder, F'{_run_name}.csv'),
        ClassificationParams.output_imgs_folder: F"{join(output_folder, results_folder, _run_name)}",
        ClassificationParams.show_imgs: True,
        ClassificationParams.save_prediction: True,
        LocalTrainingParams.forecasted_hours: 24,
        ClassificationParams.metrics: {'mse':mean_squared_error, 'mae':mean_absolute_error, 'r2':r2_score, 'ex_var':explained_variance_score},
        TrainingParams.config_name: _run_name,
    }
    return append_model_params(cur_config)
