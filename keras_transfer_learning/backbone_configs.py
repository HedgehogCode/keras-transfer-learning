from .config import BackboneConfig
from .backbones import unet


class UnetBackboneConfig(BackboneConfig):

    def __init__(self, args, weights):
        self._args = args
        self._weights = weights

    def create_backbone(self, inp):
        return unet.unet(**self._args)(inp)

    def load_weights(self, model):
        if self._weights is not None:
            model.load_weights(self._weights)
