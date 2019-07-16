"""General training script. Uses a config dictionary to create the model and train it on configured
data.

LIST OF TODOS:
TODO Add unet with border (head)
"""
import os

import pandas as pd

from keras import callbacks

from keras_transfer_learning import model, dataset, utils

###################################################################################################
#     TRAINING HELPERS
###################################################################################################


def _create_callbacks(conf, model_dir):
    training_callbacks = []

    def remove_old_checkpoints_fn(**kwargs):
        return utils.callbacks.RemoveOldCheckpoints(
            os.path.join('models', conf['name']), **kwargs)

    callback_fns = {
        'early_stopping': callbacks.EarlyStopping,
        'reduce_lr_on_plateau': callbacks.ReduceLROnPlateau,
        'remove_old_checkpoints': remove_old_checkpoints_fn
    }

    # Default callbackse
    training_callbacks.append(_checkpoints_callback(model_dir))
    training_callbacks.append(_tensorboard_callback(
        conf['name'], conf['training']['batch_size']))

    # Configured callbacks
    for callback in conf['training']['callbacks']:
        training_callbacks.append(
            callback_fns[callback['name']](**callback['args']))

    return training_callbacks


def _checkpoints_callback(model_dir):
    checkpoint_filename = os.path.join(
        model_dir, 'weights_{epoch:04d}_{val_loss:.4f}.h5')
    return callbacks.ModelCheckpoint(
        checkpoint_filename, save_weights_only=True)


def _tensorboard_callback(model_name, batch_size):
    tensorboard_logdir = os.path.join('.', 'logs', model_name)
    return callbacks.TensorBoard(
        tensorboard_logdir, batch_size=batch_size, write_graph=True)


###################################################################################################
#     TRAINING PROCEDURE
###################################################################################################

def train(conf: dict, epochs: int, initial_epoch: int = 0):
    print('\n\nStarting training of model {}\n'.format(conf['name']))
    print('Creating the model...')
    if initial_epoch == 0:
        load_weights = 'pretrained'
    else:
        load_weights = 'last'
    m = model.Model(conf, load_weights=load_weights, epoch=initial_epoch)
    m.model.summary(line_length=140)

    m.prepare_for_training()

    # Prepare the data generators
    print('Preparing data...')
    d = dataset.Dataset(conf)
    train_generator, val_generator = d.create_data_generators()

    # Create the callbacks
    training_callbacks = _create_callbacks(conf, m.model_dir)

    # Prepare the model directory
    if initial_epoch == 0:
        m.create_model_dir()

    # Train the model
    print('Training the model...')
    history = m.model.fit_generator(train_generator, validation_data=val_generator,
                                    epochs=epochs, initial_epoch=initial_epoch,
                                    callbacks=training_callbacks)

    # Save the history
    print('Saving the history...')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(m.model_dir, 'history.csv'))

    # Save the final weights
    print('Saving the final weights...')
    m.model.save_weights(os.path.join(m.model_dir, 'weights_final.h5'))
