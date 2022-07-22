from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from numpy.distutils.system_info import numarray_info

from AI.models.modelBuilder3D import *

def getModel_3D_Single(imgs_dims):
    filterFactor = 1
    [w, h, d] = imgs_dims

    inputs = Input((d, h, w, 1))
    conv1 = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv4)

    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv4)
    up6 = concatenate([up6, conv3])
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv6)
    up7 = concatenate([up7, conv2])
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)
    up8 = concatenate([up8, conv1])
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def getModel_3D_MultiORIGINAL(imgs_dims, last_activation='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    conv1_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    conv2_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    conv3_tra = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    conv1_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    conv2_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    conv3_cor = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    conv1_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    conv2_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    conv3_sag = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    conv4 = Conv3D(192 * filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv4)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv4)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv6)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)

    conv10 = Conv3D(1, (1, 1, 1), activation=last_activation)(conv8)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model


def getModel_3D_MultiORIGINAL_Dropout(imgs_dims, last_activation='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = Dropout(rate=0.2)(conv1_tra)
    conv1_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = Dropout(rate=0.2)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = Dropout(rate=0.2)(conv2_tra)
    conv2_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = Dropout(rate=0.2)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = Dropout(rate=0.2)(conv3_tra)
    conv3_tra = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = Dropout(rate=0.2)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = Dropout(rate=0.2)(conv1_cor)
    conv1_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = Dropout(rate=0.2)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = Dropout(rate=0.2)(conv2_cor)
    conv2_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = Dropout(rate=0.2)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = Dropout(rate=0.2)(conv3_cor)
    conv3_cor = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = Dropout(rate=0.2)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = Dropout(rate=0.2)(conv1_sag)
    conv1_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = Dropout(rate=0.2)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = Dropout(rate=0.2)(conv2_sag)
    conv2_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = Dropout(rate=0.2)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = Dropout(rate=0.2)(conv3_sag)
    conv3_sag = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = Dropout(rate=0.2)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    conv4 = Conv3D(192 * filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Dropout(rate=0.2)(conv4)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(rate=0.2)(conv4)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv4)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(rate=0.2)(conv6)
    conv6 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(rate=0.2)(conv6)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv6)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(rate=0.2)(conv7)
    conv7 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(rate=0.2)(conv7)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(rate=0.2)(conv8)
    conv8 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(rate=0.2)(conv8)

    conv10 = Conv3D(1, (1, 1, 1), activation=last_activation)(conv8)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model


def getModel_3D_Multi_3Streams(imgs_dims, batch_normalization=True, dropout=True, last_activation='sigmoid'):
    """
    This function returns a model with a 3D multistream input and single output. It can apply batchnormalization
    and dropout to the decoding and encoding part. (we have only tested using bn and dropout on the decoding section
    because it doesn't fit in our GPU if we use both).
    :param imgs_dims:
    :param batch_normalization:
    :param dropout:
    :param last_activation:
    :return:
    """
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    conv1_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    conv2_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    conv3_tra = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    conv1_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    conv2_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    conv3_cor = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    conv1_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    conv2_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    conv3_sag = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    # conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv6 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv8 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv10 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv11)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv12 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv14 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv15)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv16 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv18 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)

    conv20 = Conv3D(1, (1, 1, 1), activation=last_activation)(conv19)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])

    return model


def getModel_3D_Multi_Dropout(imgs_dims, last_activation='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    conv1_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    conv2_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    conv3_tra = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    conv1_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    conv2_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    conv3_cor = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8 * filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    conv1_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    conv2_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    conv3_sag = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    # conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv5 = Dropout(rate=0.2)(conv5)
    conv6 = Conv3D(128 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    conv7 = Dropout(rate=0.2)(conv7)
    up6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv7)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv8 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv9 = Dropout(rate=0.2)(conv9)
    conv10 = Conv3D(64 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    conv11 = Dropout(rate=0.2)(conv11)
    up7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv11)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv12 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv13 = Dropout(rate=0.2)(conv13)
    conv14 = Conv3D(32 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    conv15 = Dropout(rate=0.2)(conv15)
    up8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv15)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv16 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv17 = Dropout(rate=0.2)(conv17)
    conv18 = Conv3D(16 * filterFactor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)
    conv19 = Dropout(rate=0.2)(conv19)

    conv20 = Conv3D(1, (1, 1, 1), activation=last_activation)(conv19)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])

    return model


def getModel_3D_OZ1(imgs_dims, last_activation='sigmoid',  start_num_filters=8, batch_normalization_encoding=False,
                    batch_normalization_decoding=True, dropout_encoding=False, dropout_decoding=True):
    [w, h, d] = imgs_dims

    fsize = 3
    start_num_filters = 16

    bn_enc = batch_normalization_encoding  # Batch normalization encoding phase
    drop_enc = dropout_encoding  # Dropout encoding phase
    bn_dec = batch_normalization_decoding # Batch normalization decoding phase
    drop_dec = dropout_decoding  # Dropout decoding phase

    #### tra branch #####
    num_filters = start_num_filters
    inputs_tra = Input((w, h, d, 1))
    conv1_tra, pool1_tra = multiple_conv_lay_3d(inputs_tra, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv2_tra, pool2_tra = multiple_conv_lay_3d(pool1_tra, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv3_tra, pool3_tra = multiple_conv_lay_3d(pool2_tra, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)

    #### cor branch #####
    num_filters = start_num_filters
    inputs_cor = Input((w, h, d, 1))
    conv1_cor, pool1_cor = multiple_conv_lay_3d(inputs_cor, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv2_cor, pool2_cor = multiple_conv_lay_3d(pool1_cor, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv3_cor, pool3_cor = multiple_conv_lay_3d(pool2_cor, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)

    #### sag branch #####
    num_filters = start_num_filters
    inputs_sag = Input((w, h, d, 1))
    conv1_sag, pool1_sag = multiple_conv_lay_3d(inputs_sag, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv2_sag, pool2_sag = multiple_conv_lay_3d(pool1_sag, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)
    num_filters = num_filters * 2
    conv3_sag, pool3_sag = multiple_conv_lay_3d(pool2_sag, num_filters, fsize, make_pool=True, batch_norm=bn_enc, dropout=drop_enc)

    # --------- Bottom ---------
    num_filters = 128
    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])  # 21^3 x 64
    conv4, _ = multiple_conv_lay_3d(merge, num_filters, fsize, make_pool=False, batch_norm=bn_dec, dropout=drop_dec)
    up6 = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv4)

    # --------- DECODING -------
    num_filters = int(num_filters / 2)
    merge2 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])   # 42^3 x 128 + 42^3 64 + 42^3 64 + 42^3 64
    conv5, _ = multiple_conv_lay_3d(merge2, num_filters, fsize, make_pool=False, batch_norm=bn_dec, dropout=drop_dec)
    up7 = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv5)

    num_filters = int(num_filters / 2)
    merge3 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])  # 84^3 x 64
    conv6, _ = multiple_conv_lay_3d(merge3, num_filters, fsize, make_pool=False, batch_norm=bn_dec, dropout=drop_dec)
    up8 = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(conv6)

    num_filters = int(num_filters / 2)
    merge4 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])
    conv7, _ = multiple_conv_lay_3d(merge4, num_filters, fsize, make_pool=False, batch_norm=bn_dec, dropout=drop_dec)

    conv10 = Conv3D(1, (1, 1, 1), activation=last_activation)(conv7)
    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv10])

    return model


def getModel_3D_Multi_Dropout(imgs_dims, last_layer='sigmoid'):
    filterFactor = 1
    [w, h, d] = imgs_dims
    # [w, h, d] = [128,128,128]
    # [w, h, d] = [168,168,168]
    #### tra branch #####
    inputs_tra = Input((w, h, d, 1))
    conv1_tra = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    conv1_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_tra)
    # conv1_tra = BatchNormalization(axis=4)(conv1_tra)
    pool1_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv1_tra)

    conv2_tra = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    conv2_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_tra)
    # conv2_tra = BatchNormalization(axis=4)(conv2_tra)
    pool2_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv2_tra)

    conv3_tra = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    conv3_tra = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_tra)
    # conv3_tra = BatchNormalization(axis=4)(conv3_tra)
    pool3_tra = MaxPooling3D(pool_size=(2, 2, 2))(conv3_tra)

    ###### cor branch #####

    inputs_cor = Input((w, h, d, 1))
    conv1_cor = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    conv1_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_cor)
    # conv1_cor = BatchNormalization(axis=4)(conv1_cor)
    pool1_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv1_cor)

    conv2_cor = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    conv2_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_cor)
    # conv2_cor = BatchNormalization(axis=4)(conv2_cor)
    pool2_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv2_cor)

    conv3_cor = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    conv3_cor = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_cor)
    # conv3_cor = BatchNormalization(axis=4)(conv3_cor)
    pool3_cor = MaxPooling3D(pool_size=(2, 2, 2))(conv3_cor)

    ###### sag branch #####

    inputs_sag = Input((w, h, d, 1))
    conv1_sag = Conv3D(8*filterFactor, (3, 3, 3), activation='relu', padding='same')(inputs_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    conv1_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv1_sag)
    # conv1_sag = BatchNormalization(axis=4)(conv1_sag)
    pool1_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv1_sag)

    conv2_sag = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool1_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    conv2_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv2_sag)
    # conv2_sag = BatchNormalization(axis=4)(conv2_sag)
    pool2_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv2_sag)

    conv3_sag = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(pool2_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    conv3_sag = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv3_sag)
    # conv3_sag = BatchNormalization(axis=4)(conv3_sag)
    pool3_sag = MaxPooling3D(pool_size=(2, 2, 2))(conv3_sag)

    merge = concatenate([pool3_tra, pool3_cor, pool3_sag])

    # conv4 = Conv3D(192*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv4 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(merge)
    conv5 = BatchNormalization(axis=4)(conv4)
    conv5 = Dropout(rate=0.2)(conv5)
    conv6 = Conv3D(128*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv7 = BatchNormalization(axis=4)(conv6)
    conv7 = Dropout(rate=0.2)(conv7)
    up6 = Conv3DTranspose(128,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv7)

    up6 = concatenate([up6, conv3_tra, conv3_cor, conv3_sag])

    conv8 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(up6)
    conv9 = BatchNormalization(axis=4)(conv8)
    conv9 = Dropout(rate=0.2)(conv9)
    conv10 = Conv3D(64*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv11 = BatchNormalization(axis=4)(conv10)
    conv11 = Dropout(rate=0.2)(conv11)
    up7 = Conv3DTranspose(64,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv11)

    up7 = concatenate([up7, conv2_tra, conv2_cor, conv2_sag])

    conv12 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(up7)
    conv13 = BatchNormalization(axis=4)(conv12)
    conv13 = Dropout(rate=0.2)(conv13)
    conv14 = Conv3D(32*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv13)
    conv15 = BatchNormalization(axis=4)(conv14)
    conv15 = Dropout(rate=0.2)(conv15)
    up8 = Conv3DTranspose(32,(2,2,2), strides = (2,2,2), activation = 'relu', padding = 'same' )(conv15)

    up8 = concatenate([up8, conv1_tra, conv1_cor, conv1_sag])

    conv16 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(up8)
    conv17 = BatchNormalization(axis=4)(conv16)
    conv17 = Dropout(rate=0.2)(conv17)
    conv18 = Conv3D(16*filterFactor, (3, 3, 3), activation='relu', padding='same')(conv17)
    conv19 = BatchNormalization(axis=4)(conv18)
    conv19 = Dropout(rate=0.2)(conv19)

    conv20 = Conv3D(1, (1, 1, 1), activation=last_layer)(conv19)

    model = Model(inputs=[inputs_tra, inputs_sag, inputs_cor], outputs=[conv20])

    return model

