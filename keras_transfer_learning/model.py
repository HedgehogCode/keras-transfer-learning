import os
import yaml
from yaml import safe_load as yaml_load
from tqdm import tqdm

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
        'segm': lambda c: segm.segm(num_classes=c['num_classes'], **c['args'])(backbone),
        'fgbg-segm': lambda c: segm.segm(num_classes=2, **c['args'])(backbone),
        'fgbg-segm-weighted': lambda c: segm.segm(num_classes=2, **c['args'])(backbone),
        'stardist': lambda c: stardist.stardist(**c['args'])(backbone),
        'classification': lambda c: classification.classification(**c['args'])(backbone)
    }[conf['head']['name']](conf['head'])


def _prepare_model(conf, model):
    return {
        'segm': segm.prepare_for_training,
        'fgbg-segm': segm.prepare_for_training,
        'fgbg-segm-weighted': segm.prepare_for_training_fgbg_weigthed,
        'stardist': stardist.prepare_for_training,
        'classification': classification.prepare_for_training
    }[conf['head']['name']](model, **conf['head']['prepare_model_args'])


def _process_prediction(conf, pred):
    # TODO stardist: support custom prob threshold in config
    # TODO fgbg: support setting do label
    # TODO fgbg-weighted: Correct with dilation?
    return {
        'stardist': lambda: stardist.process_prediction(pred),
        'segm': lambda: segm.process_prediction_nclass(pred),
        'fgbg-segm': lambda: segm.process_prediction_fgbg(pred),
        'fgbg-segm-weighted': lambda: segm.process_prediction_fgbg(pred),
        'classification': lambda: pred
    }[conf['head']['name']]()


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
                print('Loading weights {}...'.format(weights))
                backbone_model.load_weights(weights, by_name=True)
            else:
                print('Loaded no backbone weights')

        # Create the head
        oups = _create_head(self.config, backbone)

        # Create the model
        self.model = models.Model(inputs=inp, outputs=oups)

        # Load other weights
        if load_weights == 'last':
            last_weights = utils.get_last_weights(self.model_dir, epoch=epoch)
            self.model.load_weights(last_weights, by_name=True)
        # TODO allow loading weights of one specific epoch

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

    def predict(self, data):
        """Runs inference of the model on the given data

        Arguments:
            data {list or ndarray} -- A data sample or list of data samples

        Returns:
            list -- A list of predictions
        """

        if not isinstance(data, list):
            data = [data]

        in_shape = self.model.input.shape[1:]  # - batch dimension

        preds = []
        for sample in tqdm(data):
            # Fix shape of the data
            for _ in range(len(in_shape) - len(sample.shape)):
                sample = sample[..., None]

            # TODO check the shape of the data

            pred = self.model.predict(sample[None, ...])
            if isinstance(pred, list):
                pred = tuple(p[0, ...] for p in pred)
            else:
                pred = pred[0, ...]

            preds.append(pred)

        return preds

    def process_prediction(self, preds):
        if not isinstance(preds, list):
            preds = [preds]

        processed = []
        for pred in preds:
            processed.append(_process_prediction(self.config, pred))

        return processed

    def predict_and_process(self, data):
        return self.process_prediction(self.predict(data))
