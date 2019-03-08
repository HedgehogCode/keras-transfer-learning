import keras.backend as K
import tensorflow as tf


def write_graph_to_tensorboard(log_dir):
    writer = tf.summary.FileWriter(log_dir, K.get_session().graph)
    writer.flush()
