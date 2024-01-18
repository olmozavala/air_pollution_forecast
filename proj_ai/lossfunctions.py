import tensorflow as tf

def custom_mse(y_true, y_pred):
    # return tf.reduce_mean(tf.square(y_true - y_pred))
    mask = tf.not_equal(y_true, -200)  # Create a mask for non -200 values
    masked_true = tf.boolean_mask(y_true, mask)
    masked_pred = tf.boolean_mask(y_pred, mask)
    squared_diff = tf.square(masked_true - masked_pred)
    return tf.reduce_mean(squared_diff)
