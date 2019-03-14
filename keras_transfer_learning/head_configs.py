import numpy as np

from .heads import segm
from .config import HeadConfig


class FgBgSegmHeadConfig(HeadConfig):

    def __init__(self, args, prepare_model_args):
        self._args = args
        self._prepare_model_args = prepare_model_args

    def create_head(self, backbone):
        return segm.segm(num_classes=2, **self._args)(backbone)

    def prepare_model(self, model):
        return segm.prepare_for_training(model, **self._prepare_model_args)

    def prepare_data(self, batch_x, batch_y):
        out_x = batch_x[..., None]  # TODO input with channels?
        foreground = np.array(batch_y) > 0
        background = np.logical_not(foreground)
        out_y = np.array(
            np.stack([foreground, background], axis=-1), dtype='float32')
        return out_x, out_y
