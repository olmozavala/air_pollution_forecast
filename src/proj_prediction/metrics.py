import tensorflow as tf
import tensorflow.keras.backend as K

# def compute_metric(x, y, metric_name):

def restricted_mse(y_true, y_pred):
    eps = .00001
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    temp = tf.math.ceil(y_true_f + eps)  # We make everything 1 except the original nan values, now should be 0
    y_pred_c = y_pred_f * temp  # Now we make the prediction for those stations 0, no matter what was the original value
    y_true_c = y_true_f * temp  # We make the original -1 (nans) also to 0 so that they have the same value as the pred
    return tf.math.reduce_mean(tf.math.squared_difference(y_pred_c, y_true_c))