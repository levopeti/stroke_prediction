import tensorflow as tf

from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
# tf.config.run_functions_eagerly(True)


def stroke_loss_clas(stroke_loss_factor):
    def inner_loss(y_true, y_pred):
        cce = CategoricalCrossentropy()
        cce_loss = cce(y_true, y_pred)

        stroke_loss = tf.reduce_mean(
            tf.cast(tf.math.logical_xor(tf.argmax(y_true, axis=1) == 5, tf.argmax(y_pred, axis=1) == 5), tf.float32))
        loss = stroke_loss * stroke_loss_factor + cce_loss
        return loss

    return inner_loss


def stroke_loss_reg(stroke_loss_factor):
    def inner_loss(y_true, y_pred):
        y_pred *= 6
        y_pred -= 0.5
        # y_pred = tf.cond(y_pred > 5.49, lambda: y_pred - 0.1, lambda: y_pred)
        mse = MeanSquaredError()
        mse_loss = mse(y_true, y_pred)

        stroke_loss = tf.reduce_mean(tf.cast(tf.math.logical_xor(y_true == 5, tf.round(y_pred) == 5), tf.float32))
        loss = stroke_loss * stroke_loss_factor + mse_loss
        return loss

    return inner_loss


def stroke_accuracy_clas(y_true, y_pred):
    return tf.reduce_mean(
        tf.cast(tf.math.logical_not(tf.math.logical_xor(tf.argmax(y_true, axis=1) == 5, tf.argmax(y_pred, axis=1) == 5), tf.float32)))


def stroke_accuracy_reg(y_true, y_pred):
    y_pred *= 6
    y_pred -= 0.5
    # y_pred = tf.cond(y_pred > 5.49, lambda: y_pred - 0.1, lambda: y_pred)
    return tf.reduce_mean(tf.cast(tf.math.logical_not(tf.math.logical_xor(y_true == 5, tf.round(y_pred) == 5)), tf.float32))

# TODO sens/spec

