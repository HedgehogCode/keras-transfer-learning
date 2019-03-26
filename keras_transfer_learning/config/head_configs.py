from abc import abstractmethod

import numpy as np

from stardist.utils import edt_prob, star_dist

from .config_holder import ConfigHolder
from ..heads import segm, stardist, classification


# Abstract definition
class HeadConfig(ConfigHolder):

    @abstractmethod
    def create_head(self, backbone):
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model):
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self, batch_x, batch_y):
        raise NotImplementedError


# Static config parser

def get_config(conf) -> HeadConfig:
    if conf['name'] == 'fgbg-segm':
        return FgBgSegmHeadConfig(conf['args'], conf['prepare_model_args'])
    if conf['name'] == 'stardist':
        return StarDistHeadConfig(conf['args'], conf['prepare_model_args'])
    if conf['name'] == 'classification':
        return ClassificationHeadConfig(conf['args'], conf['prepare_model_args'])
    raise NotImplementedError(
        'The head {} is not implemented.'.format(conf['name']))


# Implementations

class FgBgSegmHeadConfig(HeadConfig):

    def __init__(self, args, prepare_model_args):
        self.args = args
        self.prepare_model_args = prepare_model_args

    def create_head(self, backbone):
        return segm.segm(num_classes=2, **self.args)(backbone)

    def prepare_model(self, model):
        return segm.prepare_for_training(model, **self.prepare_model_args)

    def prepare_data(self, batch_x, batch_y):
        out_x = batch_x[..., None]  # TODO input with channels?
        foreground = np.array(batch_y) > 0
        background = np.logical_not(foreground)
        out_y = np.array(
            np.stack([foreground, background], axis=-1), dtype='float32')
        return out_x, out_y

    def get_as_dict(self):
        return {
            'name': 'fgbg-segm',
            'args': self.args,
            'prepare_model_args': self.prepare_model_args
        }


class StarDistHeadConfig(HeadConfig):

    def __init__(self, args, prepare_model_args):
        self.args = args
        self.prepare_model_args = prepare_model_args

    def create_head(self, backbone):
        return stardist.stardist(**self.args)(backbone)

    def prepare_model(self, model):
        return stardist.prepare_for_training(model, **self.prepare_model_args)

    def prepare_data(self, batch_x, batch_y):
        prob = np.stack([edt_prob(lbl) for lbl in batch_y])[..., None]
        dist = np.stack([star_dist(lbl, self.args['n_rays'])
                         for lbl in batch_y])
        dist_mask = prob
        img = (np.array(batch_x, dtype='float32') / 255)[..., None]
        return [img, dist_mask], [prob, dist]

    def get_as_dict(self):
        return {
            'name': 'stardist',
            'args': self.args,
            'prepare_model_args': self.prepare_model_args
        }


class ClassificationHeadConfig(HeadConfig):

    def __init__(self, args, prepare_model_args):
        self.args = args
        self.prepare_model_args = prepare_model_args

    def create_head(self, backbone):
        return classification.classification(**self.args)(backbone)

    def prepare_model(self, model):
        return stardist.prepare_for_training(model, **self.prepare_model_args)

    def prepare_data(self, batch_x, batch_y):
        return np.array(batch_x), np.array(batch_y)

    def get_as_dict(self):
        return {
            'name': 'classification',
            'args': self.args,
            'prepare_model_args': self.prepare_model_args
        }
