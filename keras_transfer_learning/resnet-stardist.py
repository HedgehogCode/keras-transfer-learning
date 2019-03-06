# TODO split into multiple reusable modules

from keras import layers
from keras import models
import keras.backend as K

from utils.tensorboard import write_graph_to_tensorboard
from resnext import resnext_101

# Build the model
inp = layers.Input(shape=(224, 224, 3))
oup = resnext_101(inp)

m = models.Model(inp, oup)

# Try to load the weights
m.load_weights('/home/bw/Downloads/resnext101_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

# Print a summary and write to tensorboard
m.summary()
write_graph_to_tensorboard(log_dir='models/my-resnet50-graph/')
