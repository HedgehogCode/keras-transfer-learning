import os
import re
import glob

from keras import callbacks

from keras_transfer_learning import utils


class RemoveOldCheckpoints(callbacks.Callback):

    def __init__(self, file_path, keep_latest=5, keep_period=5):
        super(RemoveOldCheckpoints, self).__init__()
        self._file_path = file_path
        self._keep_latest = keep_latest
        self._keep_period = keep_period

        # Regex for getting the epoch
        self._epoch_regex = re.compile(utils.utils.WEIGHTS_FILE_REGEX)

    def on_epoch_end(self, epoch, logs=None):
        # All save weights
        weights_files = sorted(glob.glob(os.path.join(
            self._file_path, 'weights_[0-9]*_*.h5')))

        # Numbers of epochs to keep
        next_epoch = epoch + 1
        keep_epochs = set(range(1, next_epoch, self._keep_period)) | \
            set(range(next_epoch - self._keep_latest, next_epoch))

        for weights_file in weights_files:
            matches = self._epoch_regex.match(weights_file)
            if matches is None:
                # Something is wrong with this weights file (ignore)
                continue

            weights_epoch = int(matches.group(1))
            if weights_epoch not in keep_epochs:
                # Delete the file
                os.remove(weights_file)
