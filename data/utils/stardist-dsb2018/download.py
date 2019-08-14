#!/usr/bin/env python

"""Downloads and extracts the subset of the 2018 Data Science Bowl dataset that was used in the
StarDist paper (Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. (2018). Cell Detection with
Star-convex Polygons. ArXiv Preprint ArXiv:1806.03535.).
The dataset was released to the GitHub repository: https://github.com/mpicbg-csbd/stardist
"""

import os
import sys
import shutil
import tempfile

from csbdeep.utils import download_and_extract_zip_file

def main():
    download_url = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip'
    target_dir = os.path.join('..', '..', 'stardist-dsb2018')

    # Create temp dir
    tempdir_obj = tempfile.TemporaryDirectory()
    tempdir = tempdir_obj.name

    # Download and extract to temp
    download_and_extract_zip_file(url=download_url, targetdir=tempdir)

    # Move to target
    shutil.move(os.path.join(tempdir, 'dsb2018'), target_dir)

    # Delete the temp dir
    tempdir_obj.cleanup()


if __name__ == '__main__':
    sys.exit(main())
