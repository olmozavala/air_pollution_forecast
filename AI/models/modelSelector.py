from ai_common.constants.AI_params import *
import AI.models.modelBuilder3D as model_builder_3d
import AI.models.modelBuilder2D as model_builder_2d
import AI.models.modelBuilder1D as model_builder_1d
from tensorflow.keras.layers import Input

def select_3d_model(model_params):
    model_type = model_params[ModelParams.MODEL]
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    # Makes a 3D-Unet with three input streams
    if model_type == AiModels.UNET_3D_3_STREAMS or model_type == AiModels.UNET_3D_SINGLE:
        # Reading configuration
        batch_normalization_dec = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout_dec = model_params[ModelParams.DROPOUT]
        start_num_filters = model_params[ModelParams.START_NUM_FILTERS]
        number_levels = model_params[ModelParams.NUMBER_LEVELS]
        filter_size = model_params[ModelParams.FILTER_SIZE]
        # Setting the proper inputs
        if model_type == AiModels.UNET_3D_3_STREAMS:
            inputs = [Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1)),
                      Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1)),
                      Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1))]
        elif model_type == AiModels.UNET_3D_SINGLE:
            inputs = [Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1))]
        # Building the model
        model = model_builder_3d.make_multistream_3d_unet(inputs, num_filters=start_num_filters,
                                                          filter_size=filter_size,
                                                          num_levels=number_levels,
                                                          batch_norm_encoding=False,
                                                          batch_norm_decoding=batch_normalization_dec,
                                                          dropout_encoding=False,
                                                          dropout_decodign=dropout_dec)
    elif (model_type == AiModels.HALF_UNET_3D_CLASSIFICATION_3_STREAMS or
          model_type == AiModels.HALF_UNET_3D_CLASSIFICATION_SINGLE_STREAM):
        # Reading configuration
        batch_normalization = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout = model_params[ModelParams.DROPOUT]
        start_num_filters = model_params[ModelParams.START_NUM_FILTERS]
        number_levels = model_params[ModelParams.NUMBER_LEVELS]
        filter_size = model_params[ModelParams.FILTER_SIZE]
        size_last_layer = model_params[ModelParams.NUMBER_OF_OUTPUT_CLASSES]
        dense_layers = model_params[ModelParams.NUMBER_DENSE_LAYERS]
        # Setting the proper inputs
        if model_type == AiModels.HALF_UNET_3D_CLASSIFICATION_3_STREAMS:
            inputs = [Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1)),
                      Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1)),
                      Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1))]
        elif model_type == AiModels.UNET_3D_SINGLE:
            inputs = [Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1))]
        # Building the model
        model = model_builder_3d.make_multistream_3d_half_unet_for_classification(inputs,
                                                                                  num_filters=start_num_filters,
                                                                                  number_of_dense_layers=dense_layers,
                                                                                  size_last_layer=size_last_layer,
                                                                                  filter_size=filter_size,
                                                                                  num_levels=number_levels,
                                                                                  batch_norm=batch_normalization,
                                                                                  dropout=dropout)
    else:
        raise Exception(F"The specified model doesn't have a configuration: {model_type.value}")

    return model


def select_2d_model(model_params):
    model_type = model_params[ModelParams.MODEL]
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    # Makes a 3D-Unet with three input streams
    if model_type == AiModels.HALF_UNET_2D_SINGLE_STREAM_CLASSIFICATION:
        # Reading configuration
        batch_normalization = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout = model_params[ModelParams.DROPOUT]
        start_num_filters = model_params[ModelParams.START_NUM_FILTERS]
        number_levels = model_params[ModelParams.NUMBER_LEVELS]
        filter_size = model_params[ModelParams.FILTER_SIZE]
        size_last_layer = model_params[ModelParams.NUMBER_OF_OUTPUT_CLASSES]
        dense_layers = model_params[ModelParams.NUMBER_DENSE_LAYERS]
        # Setting the proper inputs
        inputs = [Input((nn_input_size[0], nn_input_size[1], 1))]
        # Building the model
        model = model_builder_2d.make_multistream_2d_half_unet_for_classification(
            inputs, num_filters=start_num_filters,
            number_of_dense_layers=dense_layers,
            size_last_layer=size_last_layer,
            filter_size=filter_size,
            num_levels=number_levels,
            batch_norm=batch_normalization,
            dropout=dropout)
        # Makes a 3D-Unet with three input streams
    elif model_type == AiModels.UNET_2D_SINGLE:
        # Reading configuration
        batch_normalization = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout = model_params[ModelParams.DROPOUT]
        start_num_filters = model_params[ModelParams.START_NUM_FILTERS]
        number_levels = model_params[ModelParams.NUMBER_LEVELS]
        filter_size = model_params[ModelParams.FILTER_SIZE]
        # Setting the proper inputs
        inputs = [Input((nn_input_size[0], nn_input_size[1], 1))]
        # Building the model
        model = model_builder_2d.make_multistream_2d_unet(
            inputs,
            num_filters=start_num_filters,
            filter_size=filter_size,
            num_levels=number_levels,
            batch_norm_encoding=batch_normalization,
            batch_norm_decoding=batch_normalization,
            dropout_encoding=dropout,
            dropout_decodign=dropout)

    else:
        raise Exception(F"The specified model doesn't have a configuration: {model_type.value}")

    return model


def select_1d_model(model_params):
    model_type = model_params[ModelParams.MODEL]
    # Makes a 1D-Encoder
    if model_type == AiModels.ML_PERCEPTRON:
        # Reading configuration
        batch_normalization = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout = model_params[ModelParams.DROPOUT]
        input_size = model_params[ModelParams.INPUT_SIZE]
        number_hidden_layers = model_params[ModelParams.HIDDEN_LAYERS]
        cells_per_hidden_layer = model_params[ModelParams.CELLS_PER_HIDDEN_LAYER]
        output_layer_size= model_params[ModelParams.NUMBER_OF_OUTPUT_CLASSES]
        # Setting the proper inputs
        inputs = Input(shape=(input_size,))
        # Building the model
        model = model_builder_1d.single_multlayer_perceptron( inputs, number_hidden_layers,
                                                              cells_per_hidden_layer,
                                                              output_layer_size,
                                                              dropout=dropout,
                                                              batch_norm=batch_normalization)
    else:
        raise Exception(F"The specified model doesn't have a configuration: {model_type.value}")

    return model
