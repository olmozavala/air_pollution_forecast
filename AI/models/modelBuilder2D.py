from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def single_conv_lay_2d(input, num_filters, filter_size, batch_norm=False, dropout=False, activation='relu'):
    """ Adds a single convolutional layer with the specifiec parameters. It can add batch normalization and dropout
        :param input: The layer to use as input
        :param num_filters: The number of filters to use
        :param filter_size:  Size of the filters
        :param batch_norm: If we want to use batch normalization after the CNN
        :param dropout: If we want to use dropout after the CNN
        :return:
    """
    conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(input)
    # Adding batch normalization
    if batch_norm :
        # Default values: (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        #                  beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
        #                  moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
        #                  beta_constraint=None, gamma_constraint=None)
        next = BatchNormalization()(conv1)  # Important, which axis?
    else:
        next = conv1
    # Adding dropout
    if dropout:
        final = Dropout(rate=0.2)(next)
    else:
        final = next
    return final


def multiple_conv_lay_2d(input, num_filters, filter_size, make_pool=True, batch_norm=False,
                         dropout=False, tot_layers=2):
    """ Adds a N convolutional layers followed by a max pool
        :param tot_layers: how many layers we want to create
        :param input: The layer to use as input
        :param num_filters: The number of filters to use
        :param filter_size:  Size of the filters
        :param batch_norm: If we want to use batch normalization after the CNN
        :param dropout: If we want to use dropout after the CNN
    :return:
    """
    c_input = input
    for c_layer_idx in range(tot_layers):
        c_input = single_conv_lay_2d(c_input, num_filters, filter_size, batch_norm=batch_norm, dropout=dropout)

    if make_pool:
        maxpool = MaxPooling2D(pool_size=(2, 2))(c_input)
    else:
        maxpool = []

    return c_input, maxpool


def make_multistream_2d_unet(inputs, num_filters=8, filter_size=3, num_levels=3,
                             batch_norm_encoding=False,
                             batch_norm_decoding=True,
                             dropout_encoding=False,
                             dropout_decodign=True):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param batch_norm_encoding: Indicates if we are using batch normalization in the encoding phase
    :param batch_norm_decoding: Indicates if we are using batch normalization in the decoding phase
    :param dropout_encoding: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm_encoding,
                                                  dropout=dropout_encoding)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
    print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
    else: # This is the single stream case (Default 3D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]

    print(F'Merged size: {merged.shape}')

    # Convoulutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm_decoding,
                                          dropout=dropout_decodign)
    print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")
    conv_t = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv_p)
    print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")


    print("\n ------------- DECODING PATH ----------------")
    for level in range(1,num_levels+1):
        bottom_up_level = num_levels-level

        # print(F" Concatenating {conv_t.shape} with previous convs {streams[0]['convs'][bottom_up_level].shape} (each)")
        merged_temp = [conv_t]
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['convs'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')

        filters = num_filters * (2 ** (bottom_up_level))
        [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")

        if level != (num_levels):
            conv_t = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv_p)
            print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")

    last_conv = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model


def make_multistream_2d_half_unet_for_classification(inputs, num_filters=8, filter_size=3, num_levels=3,
                                                     size_last_layer=10,
                                                     number_of_dense_layers=2,
                                                     batch_norm=True,
                                                     dropout=True):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param batch_norm: Indicates if we are using batch normalization in the encoding phase
    :param dropout: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm,
                                                  dropout=dropout)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
    print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
    else: # This is the single stream case (Default 3D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]

    print(F'Merged size: {merged.shape}')
    # Convolutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm,
                                          dropout=dropout)
    # Dense layers
    dense_lay = Flatten()(conv_p)  # Flattens all the neurons
    print(F"Size of flatten: {dense_lay.shape} ")
    units = num_filters * (2 ** (num_levels))
    for cur_dense_layer in range(number_of_dense_layers-1):
        print(F"Neurons for dense layer {units} ")
        dense_lay = Dense(filters, activation='relu')(dense_lay)

    final_lay = Dense(size_last_layer, activation='softmax')(dense_lay)
    print(F"Final number of units: {size_last_layer}")

    model = Model(inputs=inputs, outputs=[final_lay])
    return model
