from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def single_multlayer_perceptron(input, number_hidden_layers, cells_per_hidden_layer, output_layer_size,
                                batch_norm=False, dropout=False,
                                activation_hidden='relu',
                                activation_output='sigmoid'):
    """ Creates a classic multilayer perceptron with the specifiec parameters. It can add batch normalization and dropout
        :param input: The layer to use as input
        :param number_hidden_layers: Number of hidden layers
        :param cells_per_hidden_layer:
        :param output_layer_size:
        :param batch_norm: If we want to use batch normalization after the CNN
        :param dropout: If we want to use dropout after the CNN
        :return:
    """
    dense_cur = Dense(cells_per_hidden_layer[0], activation=activation_hidden, name=F'input_layer')(input)
    for cur_hid_layer in range(1,number_hidden_layers):
        dense_cur = Dense(cells_per_hidden_layer[cur_hid_layer], activation=activation_hidden, name=F'hidden_{cur_hid_layer}')(dense_cur)
        # Adding batch normalization
        if batch_norm :
            dense_cur = BatchNormalization()(dense_cur)  # Important, which axis?
        # Adding dropout
        if dropout:
            dense_cur = Dropout(rate=0.2)(dense_cur)

    output_layer = Dense(output_layer_size, activation=activation_output, name="output_layer")(dense_cur)
    model = Model(input, output_layer)
    return model

