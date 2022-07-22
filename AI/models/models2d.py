from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from numpy.distutils.system_info import numarray_info

from models.modelUtils import *

def getModel_2D_MultiBatch(imgs_dims, last_layer='sigmoid'):
    filterFactor = 1
    [w, h] = imgs_dims
    filterSize = (3, 3)
    poolSize = (2, 2)
    #### tra branch #####
    inputs_tra = Input((w, h, 1))
    conv1_tra = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_tra)
    conv1_tra = BatchNormalization(axis=3)(conv1_tra)
    conv1_tra = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_tra)
    conv1_tra = BatchNormalization(axis=3)(conv1_tra)
    pool1_tra = MaxPooling2D(pool_size=poolSize)(conv1_tra)

    conv2_tra = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_tra)
    conv2_tra = BatchNormalization(axis=3)(conv2_tra)
    conv2_tra = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_tra)
    conv2_tra = BatchNormalization(axis=3)(conv2_tra)
    pool2_tra = MaxPooling2D(pool_size=poolSize)(conv2_tra)

    conv3_tra = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_tra)
    conv3_tra = BatchNormalization(axis=3)(conv3_tra)
    conv3_tra = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_tra)
    conv3_tra = BatchNormalization(axis=3)(conv3_tra)
    pool3_tra = MaxPooling2D(pool_size=poolSize)(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, 1))
    conv1_cor = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_cor)
    conv1_cor = BatchNormalization(axis=3)(conv1_cor)
    conv1_cor = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_cor)
    conv1_cor = BatchNormalization(axis=3)(conv1_cor)
    pool1_cor = MaxPooling2D(pool_size=poolSize)(conv1_cor)

    conv2_cor = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_cor)
    conv2_cor = BatchNormalization(axis=3)(conv2_cor)
    conv2_cor = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_cor)
    conv2_cor = BatchNormalization(axis=3)(conv2_cor)
    pool2_cor = MaxPooling2D(pool_size=poolSize)(conv2_cor)

    conv3_cor = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_cor)
    conv3_cor = BatchNormalization(axis=3)(conv3_cor)
    conv3_cor = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_cor)
    conv3_cor = BatchNormalization(axis=3)(conv3_cor)
    pool3_cor = MaxPooling2D(pool_size=poolSize)(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w,  h, 1))
    conv1_sag = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_sag)
    conv1_sag = BatchNormalization(axis=3)(conv1_sag)
    conv1_sag = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_sag)
    conv1_sag = BatchNormalization(axis=3)(conv1_sag)
    pool1_sag = MaxPooling2D(pool_size=poolSize)(conv1_sag)

    conv2_sag = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_sag)
    conv2_sag = BatchNormalization(axis=3)(conv2_sag)
    conv2_sag = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_sag)
    conv2_sag = BatchNormalization(axis=3)(conv2_sag)
    pool2_sag = MaxPooling2D(pool_size=poolSize)(conv2_sag)

    conv3_sag = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_sag)
    conv3_sag = BatchNormalization(axis=3)(conv3_sag)
    conv3_sag = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_sag)
    conv3_sag = BatchNormalization(axis=3)(conv3_sag)
    pool3_sag = MaxPooling2D(pool_size=poolSize)(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    conv4 = Conv2D(192*filterFactor, filterSize, activation='relu', padding='same')(merge)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(128*filterFactor, filterSize, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    up6 = Conv2DTranspose(128,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv4)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv6 = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    up7 = Conv2DTranspose(64,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv6)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv7 = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    up8 = Conv2DTranspose(32,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv7)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv8 = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)

    conv10 = Conv2D(1, (1, 1), activation=last_layer)(conv8)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model

def getModel_2D_Multi(imgs_dims, last_layer='sigmoid'):
    filterFactor = 1
    [w, h] = imgs_dims
    filterSize = (3, 3)
    poolSize = (2, 2)
    #### tra branch #####
    inputs_tra = Input((w, h, 1))
    conv1_tra = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_tra)
    conv1_tra = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_tra)
    pool1_tra = MaxPooling2D(pool_size=poolSize)(conv1_tra)

    conv2_tra = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_tra)
    conv2_tra = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_tra)
    pool2_tra = MaxPooling2D(pool_size=poolSize)(conv2_tra)

    conv3_tra = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_tra)
    conv3_tra = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_tra)
    pool3_tra = MaxPooling2D(pool_size=poolSize)(conv3_tra)

    ###### adc branch #####
    inputs_cor = Input((w, h, 1))
    conv1_cor = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_cor)
    conv1_cor = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_cor)
    pool1_cor = MaxPooling2D(pool_size=poolSize)(conv1_cor)

    conv2_cor = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_cor)
    conv2_cor = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_cor)
    pool2_cor = MaxPooling2D(pool_size=poolSize)(conv2_cor)

    conv3_cor = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_cor)
    conv3_cor = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_cor)
    pool3_cor = MaxPooling2D(pool_size=poolSize)(conv3_cor)

    ###### bval branch #####

    inputs_sag = Input((w,  h, 1))
    conv1_sag = Conv2D(8*filterFactor, filterSize, activation='relu', padding='same')(inputs_sag)
    conv1_sag = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv1_sag)
    pool1_sag = MaxPooling2D(pool_size=poolSize)(conv1_sag)

    conv2_sag = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(pool1_sag)
    conv2_sag = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv2_sag)
    pool2_sag = MaxPooling2D(pool_size=poolSize)(conv2_sag)

    conv3_sag = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(pool2_sag)
    conv3_sag = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv3_sag)
    pool3_sag = MaxPooling2D(pool_size=poolSize)(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    conv4 = Conv2D(192*filterFactor, filterSize, activation='relu', padding='same')(merge)
    conv4 = Conv2D(128*filterFactor, filterSize, activation='relu', padding='same')(conv4)
    up6 = Conv2DTranspose(128,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv4)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv6 = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(up6)
    conv6 = Conv2D(64*filterFactor, filterSize, activation='relu', padding='same')(conv6)
    up7 = Conv2DTranspose(64,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv6)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv7 = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(up7)
    conv7 = Conv2D(32*filterFactor, filterSize, activation='relu', padding='same')(conv7)
    up8 = Conv2DTranspose(32,(2,2), strides = (2,2), activation = 'relu', padding = 'same' )(conv7)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv8 = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(up8)
    conv8 = Conv2D(16*filterFactor, filterSize, activation='relu', padding='same')(conv8)

    conv10 = Conv2D(1, (1, 1), activation=last_layer)(conv8)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model
