import os
import yaml
from yaml import safe_load as yaml_load

from keras import layers, models

from keras_transfer_learning.utils import utils
from keras_transfer_learning.backbones import unet, convnet
from keras_transfer_learning.heads import segm, stardist, classification


###################################################################################################
#     BACKBONE HELPERS
###################################################################################################

def _create_backbone(conf, inp):
    return {
        'unet': lambda: unet.unet(**conf['backbone']['args'])(inp),
        'unet-csbdeep': lambda: unet.unet_csbdeep(**conf['backbone']['args'])(inp),
        'convnet': lambda: convnet.convnet(**conf['backbone']['args'])(inp)
    }[conf['backbone']['name']]()


###################################################################################################
#     HEAD HELPERS
###################################################################################################

def _create_head(conf, backbone):
    return {
        'fgbg-segm': lambda: segm.segm(num_classes=2, **conf['head']['args'])(backbone),
        'stardist': lambda: stardist.stardist(**conf['head']['args'])(backbone),
        'classification': lambda: classification.classification(**conf['head']['args'])(backbone)
    }[conf['head']['name']]()


def _prepare_model(conf, model):
    return {
        'fgbg-segm': segm.prepare_for_training,
        'stardist': stardist.prepare_for_training,
        'classification': classification.prepare_for_training
    }[conf['head']['name']](model, **conf['head']['prepare_model_args'])


###################################################################################################
#     MODEL CLASS
###################################################################################################

class Model:

    def __init__(self, config=None, model_dir=None, load_weights=None, epoch=None):
        if config is None and model_dir is None:
            raise ValueError(
                'Either the model directory or config must be given')

        # Set the config
        if config is None:
            with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
                self.config = yaml_load(f)
        else:
            self.config = config

        # Set the model directory
        if model_dir is None:
            self.model_dir = os.path.join('.', 'models', self.config['name'])
        else:
            self.model_dir = model_dir

        # Create the input
        inp = layers.Input(self.config['input_shape'])

        # Create the backbone
        backbone = _create_backbone(self.config, inp)

        # Load pretrained weights
        if load_weights == 'pretrained':
            backbone_model = models.Model(inputs=inp, outputs=backbone)
            weights = self.config['backbone']['weights']
            if weights is not None:
                backbone_model.load_weights(weights, by_name=True)

        # Create the head
        oups = _create_head(self.config, backbone)

        # Create the model
        self.model = models.Model(inputs=inp, outputs=oups)

        # Load other weights
        if load_weights == 'last':
            last_weights = utils.get_last_weights(self.model_dir, epoch=epoch)
            self.model.load_weights(last_weights, by_name=True)

    def prepare_for_training(self):
        self.model = _prepare_model(self.config, self.model)

    def create_model_dir(self):
        if os.path.exists(self.model_dir):
            raise ValueError(
                "A model with the name {} already exists.".format(self.config['name']))
        os.makedirs(self.model_dir)

        # Save the config
        with open(os.path.join(self.model_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
