# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

# TODO add license
# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='keras-transfer-learning',
    version='0.0.1',
    description='Package for tranfer learning using Keras',
    long_description=readme,
    author='Benjamin Wilhelm',
    author_email='benjamin@b-wilhelm.de',
    # url='https://gitlab.com/bewilhelm/keras-transfer-learning', TODO publish on GitHub
    # license=license, TODO add license
    packages=find_packages(exclude=('tests', 'docs'))
)

