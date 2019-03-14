import keras

from .config_holder import ConfigHolder


CALLBACK_FNS = {
    'early_stopping': keras.callbacks.EarlyStopping,
    'reduce_lr_on_plateau': keras.callbacks.ReduceLROnPlateau
}


# General implementation

class TrainingConfig(ConfigHolder):

    def __init__(self, batch_size: int, callbacks: list):
        self.batch_size = batch_size
        self.callbacks = callbacks

    def create_callbacks(self):
        created = []
        for callback in self.callbacks:
            created.append(CALLBACK_FNS[callback['name']](**callback['args']))
        return created

    def get_as_dict(self):
        return {
            'batch_size': self.batch_size,
            'callbacks': self.callbacks
        }


# Static config parser

def get_config(conf) -> TrainingConfig:
    return TrainingConfig(conf['batch_size'], conf['callbacks'])
