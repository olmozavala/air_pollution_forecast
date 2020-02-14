from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import *
from preproc.contour_smoothing import getContourFromMask
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

def real_dice_coef(y_true, y_pred):
    smooth = 0.01
    y_true_f = K.flatten(y_true)
    # y_true_f = tf.minimum(K.flatten(y_true), 1)
    y_pred_f = tf.minimum(K.flatten(y_pred), 1)
    intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / ( K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
    # The squared is not necessary when the GT is 0 or 1 and prediction between 0 and 1
    return (2. * intersection + smooth) / ( K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    smooth = 0.01
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #y_pred_f = tf.minimum(K.flatten(y_pred), 1)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / ( K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mse(y_true, y_pred, smooth=1.0):
    eps = .001
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # y_true_f MUST have 2 for lesion and 1 for prostate
    temp = y_true_f/2.0 # temp have 1 for lesion and .5 for prostate
    y_pred_c = y_pred_f * tf.ceil(temp - eps) # Make 0 the prediction outside the prostate
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + eps)
    return tf.squared_difference(y_pred_c, y_true_c)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def numpy_dice(y_true, y_pred, smooth=1.0):

    intersection = y_true.flatten()*y_pred.flatten()

    return (2. * intersection.sum() + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def numpy_dice_copy(y_true, y_pred):
    smooth = 0.01
    y_true_f = y_true.flatten()
    y_pred_f = np.minimum(y_pred.flatten(),1)
    intersection = y_true_f * y_pred_f
    return (2. * intersection.sum() + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def numpy_hausdorff(y_true_orig, y_pred_orig):
    ''' Computes the maximum 2D Hausdorff distance from every slice'''
    y_true = getContourFromMask(y_true_orig)
    y_pred = getContourFromMask(y_pred_orig)
    xshape = y_true.shape
    temp_y_true = y_true.flatten()
    temp_y_pred = y_pred.flatten()
    xy_pos_y_true = np.unravel_index(np.where(temp_y_true > 0)[0], xshape)
    xy_pos_y_pred = np.unravel_index(np.where(temp_y_pred > 0)[0], xshape)

    try:
        hausdorff = directed_hausdorff(xy_pos_y_true, xy_pos_y_pred)
    except Exception as e:
        print(F"---------------Failed Hausdorff error: {e} ----------------")

    return hausdorff[0]

############################ 2D LESION ONLY #################################

def real_dice_coef_lesion(y_true, y_pred, smooth=1.0):
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Take into account only the prostate area
    # Multiply by 0 everything outside the prostate
    temp = y_true_f/2.0 # temp have 1 for lesion and .5 for prostate
    y_pred_c = y_pred_f * tf.ceil(temp - .001)
    y_pred_c = tf.minimum(K.flatten(y_pred_c), 1)
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + .001)
    intersection = K.sum(y_true_c * y_pred_c)
    return (2. * intersection + smooth) / ( K.sum(y_true_c) + K.sum(y_pred_c) + smooth)

def dice_coef_lesion_loss(y_true, y_pred):
    return -dice_coef_lesion(y_true, y_pred)

def dice_coef_lesion(y_true, y_pred, smooth=1.0):
    eps = .001
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # y_true_f MUST have 2 for lesion and 1 for prostate
    temp = y_true_f/2.0 # temp have 1 for lesion and .5 for prostate

    y_pred_c = y_pred_f * tf.ceil(temp - eps) # Make 0 the prediction outside the prostate
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + eps)
    intersection = K.sum(y_true_c * y_pred_c)
    return (2. * intersection + smooth) / ( K.sum(y_true_c) + K.sum(y_pred_c) + smooth)
