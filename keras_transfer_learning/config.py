from argparse import Namespace
from abc import ABC, abstractmethod

import keras

CALLBACK_FNS = {
    'early_stopping': keras.callbacks.EarlyStopping,
    'reduce_lr_on_plateau': keras.callbacks.ReduceLROnPlateau
}


class BackboneConfig(ABC, Namespace):

    @abstractmethod
    def create_backbone(self, inp):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model):
        raise NotImplementedError


class HeadConfig(ABC, Namespace):

    @abstractmethod
    def create_head(self, backbone):
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model):
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self, batch_x, batch_y):
        raise NotImplementedError


class TrainingConfig(Namespace):

    def __init__(self, batch_size: int, callbacks: list):
        Namespace.__init__(self, batch_size=batch_size, callbacks=callbacks)
        self.batch_size = batch_size
        self.callbacks = callbacks

    def create_callbacks(self):
        created = []
        for callback in self.callbacks:
            created.append(CALLBACK_FNS[callback['name']](callback['args']))
        return created


class DataConfig(ABC, Namespace):

    @abstractmethod
    def load_data(self, seed):
        raise NotImplementedError

    @abstractmethod
    def create_train_datagen(self, batch_size, prepare_fn, seed):
        raise NotImplementedError

    @abstractmethod
    def create_val_datagen(self, prepare_fn):
        raise NotImplementedError


class Config(Namespace):

    def __init__(self, name: str, input_shape: tuple, backbone: BackboneConfig, head: HeadConfig,
                 training: TrainingConfig, data: DataConfig):
        Namespace.__init__(self, name=name, input_shape=input_shape,
                           backbone=backbone, head=head, training=training, data=data)
        self.name = name
        self.input_shape = input_shape
        self.backbone = backbone
        self.head = head
        self.training = training
        self.data = data
