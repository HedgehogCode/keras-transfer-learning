from abc import abstractmethod

from .config_holder import ConfigHolder
from ..backbones import unet


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
        return UnetBackboneConfig(conf['args'], conf['weights'])
    raise NotImplementedError(
        'The backbone {} is not implemented.'.format(conf['name']))


# Implementations

class UnetBackboneConfig(BackboneConfig):

    def __init__(self, args, weights):
        self.args = args
        self.weights = weights

    def create_backbone(self, inp):
        return unet.unet(**self.args)(inp)

    def load_weights(self, model):
        if self.weights is not None:
            model.load_weights(self.weights)

    def get_as_dict(self):
        return {
            'name': 'unet',
            'args': self.args,
            'weights': self.weights
        }
