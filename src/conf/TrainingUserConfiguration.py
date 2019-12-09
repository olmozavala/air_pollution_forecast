from conf.params import MergeFilesParams, LocalTrainingParams
from conf.localConstants import constants
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os

# from constants.AI_params import *
from conf.AI_params import *

output_folder = '/data/UNAM/Air_Pollution_Forecast/Data'
# output_folder = '/home/olmozavala/REMOTE_PROJECTS/OUTPUT'

# =================================== TRAINING ===================================
# ----------------------------- UM -----------------------------------
_run_name = F'Relu_Sigmoid'  # Name of the model, for training and classification
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: False,
        ModelParams.HIDDEN_LAYERS: 3,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [300, 200, 100],
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 1,
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
        LocalTrainingParams.station: "MER",
        # LocalTrainingParams.station: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
        #  ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
        #  ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
        #  ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
        #  , "XCH"],
        # LocalTrainingParams.pollutant: [F"cont_{x}" for x in ["otres"]],
        LocalTrainingParams.pollutant: "cont_otres",
        LocalTrainingParams.start_date: '2017-01-01',
        LocalTrainingParams.end_date: '2019-12-31',
        LocalTrainingParams.forecasted_hours: 24,
        LocalTrainingParams.tot_num_quadrants: 16,  # 4x4
        LocalTrainingParams.num_hours_in_netcdf: 72
    }
    return append_model_params(cur_config)

def getMergeParams():
    # We are using the same parameter as the
    cur_config = getTrainingParams()
    cur_config[MergeFilesParams.output_folder] = join(output_folder, constants.merge_output_folder.value)
    return cur_config

# def get_usemodel_1d_config():
#     models_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output/Training/models'
#     model_file = 'Relu_Relu_2019_10_14_18_31-262-0.00094.hdf5'
#     cur_config = {
#         ClassificationParams.training_data_file: join(_data_folder, "SWS2forML_nowave.csv"),
#         ClassificationParams.input_folder: _data_folder,
#         ClassificationParams.output_folder: F"{join(output_folder, 'Results')}",
#         ClassificationParams.model_weights_file: join(models_folder, model_file),
#         ClassificationParams.output_file_name: 'Results.csv',
#         ClassificationParams.input_file: 'zFAST_hr.csv',
#         ClassificationParams.output_imgs_folder: F"{join(output_folder, 'Results')}",
#         ClassificationParams.show_imgs: True,
#         ClassificationParams.save_prediction: True,
#         ClassificationParams.metrics: [ClassificationMetrics.MSE],
#         TrainingParams.config_name: _run_name,
#     }
#     return append_model_params(cu