import os
import glob

import tqdm
import numpy as np
import nibabel as nib

in_data_dir = '/home/benjamin/data/kits19/data'
out_data_dir = os.path.join('..', '..', 'kits19')
test_imgs = 10

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

# List all cases
cases = sorted(glob.glob(os.path.join(in_data_dir, 'case_*')))

for i, case in tqdm.tqdm(list(enumerate(cases))):
    # File names
    img_file = os.path.join(case, 'imaging.nii.gz')
    mask_file = os.path.join(case, 'segmentation.nii.gz')

    # Loading and converting
    img = nib.load(img_file).get_fdata().astype('float32')
    mask = nib.load(mask_file).get_fdata().astype('uint8')

    # Saving as npy files
    if i < test_imgs:
        name = '{:05d}'.format(i)
        np.save(os.path.join(out_test_img_dir, name), img)
        np.save(os.path.join(out_test_mask_dir, name), mask)
    else:
        name = '{:05d}'.format(i - test_imgs)
        np.save(os.path.join(out_train_img_dir, name), img)
        np.save(os.path.join(out_train_mask_dir, name), mask)
