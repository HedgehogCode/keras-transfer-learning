from .config_holder import ConfigHolder

from .backbone_configs import BackboneConfig
from .head_configs import HeadConfig
from .training_configs import TrainingConfig
from .data_configs import DataConfig


class Config(ConfigHolder):

    def __init__(self, name: str, input_shape: tuple, backbone: BackboneConfig, head: HeadConfig,
                 training: TrainingConfig, data: DataConfig):
        self.name = name
        self.input_shape = input_shape
        self.backbone = backbone
        self.head = head
        self.training = training
        self.data = data

    def get_as_dict(self):
        return {
            'name': self.name,
            'input_shape': self.input_shape,
            'backbone': self.backbone.get_as_dict(),
            'head': self.head.get_as_dict(),
            'training': self.training.get_as_dict(),
            'data': self.data.get_as_dict()
        }
