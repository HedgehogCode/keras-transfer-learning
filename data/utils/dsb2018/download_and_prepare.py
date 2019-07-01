#!/usr/bin/env python

"""Downloads and extracts the dsb2018 dataset. Downloads a fixed version of the training dataset
from https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes
"""

import os
import sys
import tempfile
import glob
import argparse
import pathlib
from urllib import request

import tqdm

from skimage import io
import numpy as np
import pandas as pd
import tifffile
from csbdeep.utils import download_and_extract_zip_file


def main(args):
    if not args.skip_train:
        download_and_process(args.train_zip, True)
    if not args.skip_test:
        download_and_process(args.test_zip, False)


def download_and_process(zip_file, train_set):
    # Create temp dir
    tempdir_obj = tempfile.TemporaryDirectory()
    tempdir = tempdir_obj.name

    # Create the target dir
    if train_set:
        out_img_dir = os.path.join('..', '..', 'dsb2018', 'train', 'images')
        out_mask_dir = os.path.join('..', '..', 'dsb2018', 'train', 'masks')
    else:
        out_img_dir = os.path.join('..', '..', 'dsb2018', 'test', 'images')
        out_mask_dir = os.path.join('..', '..', 'dsb2018', 'test', 'masks')
    os.makedirs(out_img_dir)
    os.makedirs(out_mask_dir)

    # Download files and extract
    if zip_file is not None:
        url = pathlib.Path(zip_file.name).absolute().as_uri()
    else:
        if train_set:
            url = 'https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master.zip'
        else:
            url = 'https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip'

    download_and_extract_zip_file(url=url, targetdir=tempdir, verbose=False)

    if train_set:
        # List all images
        images = glob.glob(os.path.join(
            tempdir, 'kaggle-dsbowl-2018-dataset-fixes-master', 'stage1_train', '*'))
    else:
        # Load the ground truth for the test set
        csv_file = os.path.join(tempdir, 'stage2_solution_final.csv')
        request.urlretrieve(
            'https://data.broadinstitute.org/bbbc/BBBC038/stage2_solution_final.csv',
            csv_file)
        solution_df = pd.read_csv(csv_file)
        # Do not use ignored files
        solution_df = solution_df[solution_df['Usage'] == 'Private']

        # List the image directories
        images = [os.path.join(tempdir, idx)
                  for idx in solution_df['ImageId'].unique()]

        # Get the encoded pixels for each image
        encoded_pixels = solution_df.groupby(
            'ImageId')['EncodedPixels'].apply(list)

    for img_path in tqdm.tqdm(list(images)):
        # ------------------------ IMAGE
        # Get the image file
        identifier = img_path.rpartition(os.path.sep)[-1]
        img_file = glob.glob(os.path.join(img_path, 'images', '*.png'))

        assert len(
            img_file) == 1, 'Found more than one image for a datapoint: {}'.format(img)

        # Read the image and convert to grayscale
        img = io.imread(img_file[0])
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = np.rint(img).astype('uint8')

        # Save the image
        name = '{}.tif'.format(identifier)
        tifffile.imsave(os.path.join(out_img_dir, name), img)

        # ------------------------ MASK
        mask = np.zeros_like(img, dtype='uint16')
        if train_set:
            # Get the mask files
            mask_files = sorted(
                glob.glob(os.path.join(img_path, 'masks', '*.png')))

            # Read the mask files
            for idx, mask_file in enumerate(mask_files, 1):
                m = io.imread(mask_file)

                # Check for overlapping cells
                if np.any(np.logical_and(m > 0, mask > 0)):
                    print(
                        "WARNING: Found overlapping cells in train image {}.".format(identifier))

                mask[m > 0] = idx
        else:
            # Get the encoded pixels
            mask_pixels = encoded_pixels[identifier]

            # Loop over the cells
            for idx, pixels_str in enumerate(mask_pixels, 1):
                pixels = []
                pixel_pairs = np.array(
                    [int(idx) for idx in pixels_str.split(' ')]).reshape(-1, 2)
                # Change to zero index
                pixel_pairs = pixel_pairs - [[1, 0]]
                for pair in pixel_pairs:
                    pixels.extend(list(range(pair[0], pair[0] + pair[1])))

                rows = np.mod(pixels, mask.shape[0])
                cols = np.floor_divide(pixels, mask.shape[0])

                if np.any(mask[rows, cols] > 0):
                    print(
                        "WARNING: Found overlapping cells in test image {}.".format(identifier))

                mask[rows, cols] = idx

        # Save the mask
        tifffile.imsave(os.path.join(out_mask_dir, name), mask)

    # Cleanup temp dir
    tempdir_obj.cleanup()


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-test', action="store_true")
    parser.add_argument('--skip-train', action="store_true")
    parser.add_argument(
        '--train-zip', type=argparse.FileType('r'), required=False)
    parser.add_argument(
        '--test-zip', type=argparse.FileType('r'), required=False)

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
