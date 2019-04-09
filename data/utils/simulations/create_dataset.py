#!/usr/bin/env python

"""Create a dataset for instance segmentation training from a generated dataset.
"""

import sys
import os
import argparse
import glob
import shutil
import tifffile
import numpy as np


def copy_and_rename(files, target_dir, target_name='{:05d}.tif'):
    for idx, filename in enumerate(files):
        target_file = os.path.join(target_dir, target_name.format(idx))
        shutil.copy(filename, target_file)


def copy_and_reduce_labels(files, target_dir, target_name='{:05d}.tif'):
    for idx, filename in enumerate(files):
        target_file = os.path.join(target_dir, target_name.format(idx))
        labeling_3d = tifffile.imread(filename)
        labeling_2d = np.max(labeling_3d, axis=-1, keepdims=False)
        tifffile.imsave(target_file, labeling_2d)


def main(args):
    print(args)

    # Input directorys and files
    in_img_dir = os.path.join(args.indir, 'images_' + args.suffix)
    in_label_dir = os.path.join(args.indir, 'labels')
    in_imgs = sorted(glob.glob(os.path.join(in_img_dir, 'image_*.tif')))
    in_labels = sorted(glob.glob(os.path.join(in_label_dir, 'label_*.tif')))

    # Train and test images and labels
    num_train = args.train
    in_train_imgs = in_imgs[:num_train]
    in_train_labels = in_labels[:num_train]
    in_test_imgs = in_imgs[num_train:]
    in_test_labels = in_labels[num_train:]

    # Output directory structure
    test_dir = os.path.join(args.outdir, 'test')
    train_dir = os.path.join(args.outdir, 'train')
    test_labels_dir = os.path.join(test_dir, 'masks')
    train_labels_dir = os.path.join(train_dir, 'masks')
    test_images_dir = os.path.join(test_dir, 'images')
    train_images_dir = os.path.join(train_dir, 'images')

    os.mkdir(args.outdir)
    os.mkdir(test_dir)
    os.mkdir(train_dir)
    os.mkdir(test_labels_dir)
    os.mkdir(train_labels_dir)
    os.mkdir(test_images_dir)
    os.mkdir(train_images_dir)

    # Move images
    copy_and_rename(in_train_imgs, train_images_dir)
    copy_and_rename(in_test_imgs, test_images_dir)

    # Save labels
    copy_and_reduce_labels(in_train_labels, train_labels_dir)
    copy_and_reduce_labels(in_test_labels, test_labels_dir)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('indir', help="Input directory")
    parser.add_argument('outdir', help="Output directory")
    parser.add_argument('-s', '--suffix',
                        help="Suffix for the final image directory", type=str)
    parser.add_argument('-t', '--train',
                        help="Size of the train dataset", type=int)

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
