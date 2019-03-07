# TODO split into multiple reusable modules

from keras import layers
from keras import models
import keras.backend as K

from keras_transfer_learning.utils.tensorboard import write_graph_to_tensorboard
from keras_transfer_learning.backbones.unet import unet

# Build the model
inp = layers.Input(shape=(None, None, 1))
oup = unet()(inp)

m = models.Model(inp, oup)

# Try to load the weights
# m.load_weights('/home/bw/Downloads/resnext101_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

# Print a summary and write to tensorboard
m.summary()
write_graph_to_tensorboard(log_dir='models/my-unet-graph/')
