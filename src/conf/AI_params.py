from enum import Enum


class AiModels(Enum):
    UNET_3D_SINGLE = 3
    UNET_3D_3_STREAMS = 4
    HALF_UNET_2D_SINGLE_STREAM_CLASSIFICATION = 5
    HALF_UNET_3D_CLASSIFICATION_3_STREAMS = 6
    HALF_UNET_3D_CLASSIFICATION_SINGLE_STREAM = 7

    UNET_2D_SINGLE = 20

    ML_PERCEPTRON = 10



class ModelParams(Enum):
    DROPOUT = 1  # If we should use Dropout on the NN
    BATCH_NORMALIZATION = 2  # If we are using Batch Normalization
    MODEL = 4
    INPUT_SIZE = 7
    START_NUM_FILTERS = 8  # How many filters are we using on the first layer and level
    NUMBER_LEVELS = 9  # Number of levels in the network (works for U-Net
    FILTER_SIZE = 10  # The size of the filters in each layer (currently is the same for each layer)
    # ========= For Classification ===============
    NUMBER_DENSE_LAYERS = 11  # Used in 2D
    NUMBER_OF_OUTPUT_CLASSES = 12  # Used in 2D

    # ========= For 1D Multilayer perceptron===============
    HIDDEN_LAYERS = 13
    CELLS_PER_HIDDEN_LAYER = 14


class TrainingParams(Enum):
    # These are common training parameters
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    cases = 5  # A numpy array of the cases of interest or 'all'
    validation_percentage = 6
    test_percentage = 7
    evaluation_metrics = 10
    loss_function = 11
    batch_size = 12
    epochs = 13
    config_name = 14  # A name that allows you to identify the configuration of this training
    optimizer = 15
    data_augmentation = 16

    # ============ These parameters are for images (2D or 3D) ============
    output_imgs_folder = 3  # Where to store intermediate images
    show_imgs = 4  # If we want to display the images while are being generated (for PyCharm)

    # ============ These parameters are for segmentation trainings ============
    image_file_names = 8
    ctr_file_names = 9

    # ============ These parameters are for classification ============
    class_label_file_name = 20

    # ============ These parameters are for 1D approximation ============
    file_name = 30



class ClassificationParams(Enum):
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    output_imgs_folder = 3  # Where to store intermediate images
    output_file_name = 4
    show_imgs = 5  # If we want to display the images while are being generated (for PyCharm)
    cases = 6  # A numpy array of the cases of interest or 'all'
    save_segmented_ctrs = 7  # Boolean that indicates if we need to save the segmentations
    model_weights_file = 8  # Which model weights file are we going to use
    # Indicates that we need to resample everything to the original resolution. If that is the case
    compute_original_resolution = 9
    resampled_resolution_image_name = 55  # Itk image name of a resampled resolution (for metrics in original size)
    original_resolution_image_name = 56  # Itk image name of a original resolution (for metrics in original size)
    original_resolution_ctr_name = 57  # Itk image name of a original resolution (for metrics in original size)
    metrics = 10
    segmentation_type = 11
    compute_metrics = 12  # This means we have the GT ctrs
    output_ctr_file_names = 13  # Only used if we are computing metrics
    input_img_file_names = 15  # Name of the images to read and use as input for the NN
    save_imgs = 16  # Indicates if we want to save images from the segmented contours
    save_img_slices = 17  # IF we are saving the images, it indicates which slices to save
    save_img_planes = 18  # IF we are saving the images, it indicates which plane to save
#     ================== For Time series =============
    input_file = 19
    training_data_file= 20 # Points to the file used for training
    save_prediction = 21


class SubstractionParams(Enum):
    # This is what is being used to compute TZ. It uses two contours, compute the difference and obtain its DSC
    model_weights_file = 0
    ctr_file_name = 1


class SegmentationTypes(Enum):
    """ Types of segmentation that are preconfigured """
    PROSTATE = 'Prostate'
    PZ = 'PZ'

class ClassificationMetrics(Enum):
    DSC_3D = '3D_DSC'  # DSC in 3D
    MSE = 'MSE'

