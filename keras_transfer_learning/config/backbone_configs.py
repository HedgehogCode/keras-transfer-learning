from abc import abstractmethod

from .config_holder import ConfigHolder
from ..backbones import unet, convnet


# Abstract definition
class BackboneConfig(ConfigHolder):

    @abstractmethod
    def create_backbone(self, inp):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model):
        raise NotImplementedError


# Static config parser

def get_config(conf) -> BackboneConfig:
    if conf['name'] == 'unet':
        return UNetBackboneConfig(conf['args'], conf['weights'])
    if conf['name'] == 'unet-csbdeep':
        return UNetCSBDeepBackboneConfig(conf['args'], conf['weights'])
    if conf['name'] == 'convnet':
        return ConvNetBackboneConfig(conf['args'], conf['weights'])
    raise NotImplementedError(
        'The backbone {} is not implemented.'.format(conf['name']))


# Implementations

class UNetBackboneConfig(BackboneConfig):

    def __init__(self, args, weights):
        self.args = args
        self.weights = weights

    def create_backbone(self, inp):
        return unet.unet(**self.args)(inp)

    def load_weights(self, model):
        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

    def get_as_dict(self):
        return {
            'name': 'unet',
            'args': self.args,
            'weights': self.weights
        }


class UNetCSBDeepBackboneConfig(BackboneConfig):

    def __init__(self, args, weights):
        self.args = args
        self.weights = weights

    def create_backbone(self, inp):
        return unet.unet_csbdeep(**self.args)(inp)

    def load_weights(self, model):
        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

    def get_as_dict(self):
        return {
            'name': 'unet-csbdeep',
            'args': self.args,
            'weights': self.weights
        }


class ConvNetBackboneConfig(BackboneConfig):

    def __init__(self, args, weights):
        self.args = args
        self.weights = weights

    def create_backbone(self, inp):
        return convnet.convnet(**self.args)(inp)

    def load_weights(self, model):
        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

    def get_as_dict(self):
        return {
            'name': 'convnet',
            'args': self.args,
            'weights': self.weights
        }
