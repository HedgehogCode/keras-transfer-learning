#!/usr/bin/env python

"""Create a dsb2018 submission csv according to a configuration file.
"""

import os
import sys
import argparse
import glob
from yaml import unsafe_load as yaml_load

import numpy as np
import tqdm
from scipy import ndimage

from keras_transfer_learning import model, dataset


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configfile', help='Config file',
                        type=argparse.FileType('r'))
    args = parser.parse_args(arguments)

    # Load the config yaml
    conf = yaml_load(args.configfile)

    # Load the model
    m = model.Model(conf, load_weights='last')

    # Create dataset
    d = dataset.Dataset(conf)
    test_x, _ = d.create_test_dataset()

    # Get the image ids
    folders = sorted(glob.glob(os.path.join(
        conf['data']['data_dir'], 'test', 'images', '*.tif')))
    ids = [i.split(os.path.sep)[-1][:-4] for i in folders]

    # Run the prediction
    pred = m.predict_and_process(test_x)

    # Create the submission file
    with open(os.path.join('models', conf['name'], 'submission.csv'), 'w') as submission_file:

        # Add the header
        submission_file.write('ImageId,EncodedPixels\n')

        # Loop over images and write the encoded pixels
        for image_id, p in tqdm.tqdm(list(zip(ids, pred))):
            added_one = False
            for i in range(1, np.max(p) + 1):
                segment = p == i
                if np.any(segment):
                    enc_pixels = get_encoded_pixels(segment)
                    submission_file.write(image_id + ',' + enc_pixels + '\n')
                    added_one = True
            if not added_one:
                print("WARNING: No nuclei found for image " + image_id)

        submission_file.flush()


def get_encoded_pixels(p):
    pixels = np.reshape(p, (-1,), 'F')
    lbl, _ = ndimage.label(pixels)
    objs = ndimage.find_objects(lbl)
    ranges = [(s[0].start + 1, s[0].stop - s[0].start) for s in objs]
    return ' '.join(['{} {}'.format(start, length) for start, length in ranges])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
