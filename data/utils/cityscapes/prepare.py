import os
import glob

import tqdm
import numpy as np
from skimage import io
import tifffile

in_data_dir = '/home/benjamin/data/cityscapes'
out_data_dir = os.path.join('..', '..', 'cityscapes')

# Define out directories
out_train_dir = os.path.join(out_data_dir, 'train')
out_test_dir = os.path.join(out_data_dir, 'test')
out_train_img_dir = os.path.join(out_train_dir, 'images')
out_train_mask_dir = os.path.join(out_train_dir, 'masks')
out_test_img_dir = os.path.join(out_test_dir, 'images')
out_test_mask_dir = os.path.join(out_test_dir, 'masks')

# Create out directories
os.mkdir(out_data_dir)
os.mkdir(out_train_dir)
os.mkdir(out_test_dir)
os.mkdir(out_train_img_dir)
os.mkdir(out_train_mask_dir)
os.mkdir(out_test_img_dir)
os.mkdir(out_test_mask_dir)


def process_files(img_files, mask_files, out_img_dir, out_mask_dir):
    for i, (img_file, lbl_file) in tqdm.tqdm(list(enumerate(zip(img_files, mask_files)))):
        # Read the image and the labeling
        img = io.imread(img_file)
        mask = io.imread(lbl_file)

        # Convert the image to grayscale
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = np.rint(img).astype('uint8')

        name = '{:05d}.tif'.format(i)
        tifffile.imsave(os.path.join(out_img_dir, name), img)
        tifffile.imsave(os.path.join(out_mask_dir, name), mask)


# Training files
train_img_cities = sorted(
    glob.glob(os.path.join(in_data_dir, 'leftImg8bit', 'train', '*')))
train_img_files = [f for city in train_img_cities
                   for f in sorted(glob.glob(os.path.join(city, '*.png')))]

train_mask_cities = sorted(
    glob.glob(os.path.join(in_data_dir, 'gtFine', 'train', '*')))
train_mask_files = [f for city in train_mask_cities
                    for f in sorted(glob.glob(os.path.join(city, '*labelIds.png')))]

process_files(train_img_files, train_mask_files, out_train_img_dir, out_train_mask_dir)


# Test files
# NOTE: we use the val set from the data because the labels are provided
test_img_cities = sorted(
    glob.glob(os.path.join(in_data_dir, 'leftImg8bit', 'val', '*')))
test_img_files = [f for city in test_img_cities
                   for f in sorted(glob.glob(os.path.join(city, '*.png')))]

test_mask_cities = sorted(
    glob.glob(os.path.join(in_data_dir, 'gtFine', 'val', '*')))
test_mask_files = [f for city in test_mask_cities
                    for f in sorted(glob.glob(os.path.join(city, '*labelIds.png')))]

process_files(test_img_files, test_mask_files, out_test_img_dir, out_test_mask_dir)
