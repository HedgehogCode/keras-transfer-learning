#!/usr/bin/env python

"""Downloads and extracts the dsb2018 dataset. Downloads a fixed version of the training dataset
from https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes
"""

import os
import tempfile

from csbdeep.utils import download_and_extract_zip_file

def main():
    tempdir = tempfile.TemporaryDirectory()
    # TODO download the files, extract,...
    download_and_extract_zip_file()

    tempdir.cleanup()


if __name__ == '__main__':
    sys.exit(main())
