# Keras Transfer Learning

A project to evaluate transfer learning on different microscopic image datasets.

The project was created for the Bachelor's thesis "Evaluating the Applicability of Transfer Learning for Deep Learning Based Segmentation of Microscope Images" by Benjamin Wilhelm at the University of Konstanz.

## Overview

The **keras_transfer_learning** package contains utilities for creating, traning and retraining instance segmentation models.

The **mean_average_precision** package contains functions to compute the mean Average Precision for instance segmentation as used in the [2018 Data Science Bowl challenge](https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation) and PASCAL VOC.

The `data` directory contains the datasets. Use the utility scripts in `data/utils` to download and prepare the datasets.

The `models` direcory contains trained models.
A trained model consists of
* a configuraion file `config.yaml`,
* a training history `history.csv`,
* evaluation results (if it was evaluated) `results.csv`,
* model weights for multiple epochs `weights_{epoch}_{val_loss}.h5` and
* the final weights `weights_final.h5`.

The `configs` directory contains configuration YAML files for backbones, heads, data and traning that can be combined to a model configuration.

The root directory contains a bunch of useful scripts and jupyter notebooks.

## Getting Started

### Requirements

The recomended way is to install the requirements into a conda environment. Otherwise the requirements can be installed manually. The list of requirements can be extracted from the [environment-gpu.yaml](environment-gpu.yaml) file.

Install the requirements using conda:

With GPU support:
```
$ conda env create -f environment-gpu.yaml -n keras-transfer-learning
```

Without GPU support:
```
$ conda env create -f environment-cpu.yaml -n keras-transfer-learning
```

Before running anything the environment must be activated:
```
$ conda activate keras-transfer-learning
```

### Training a new model

The train script can be used to train new models with implemented backbone, head and traning configuration:
```
usage: train.py [-h] -b BACKBONE --head HEAD -t TRAINING -d DATA --eval EVAL
                -n NAME -i INPUT_SHAPE -e EPOCHS

Train a deep learning model according to configuration files.

optional arguments:
  -h, --help            show this help message and exit
  -b BACKBONE, --backbone BACKBONE
                        Backbone config file
  --head HEAD           Head config file
  -t TRAINING, --training TRAINING
                        Training config file
  -d DATA, --data DATA  Data config file
  --eval EVAL           Evaluation config file
  -n NAME, --name NAME  Name of the model
  -i INPUT_SHAPE, --input_shape INPUT_SHAPE
                        Input shape of the model (yaml)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
```

### Reproducting the results

Run the `run_experiments.py` script to train all models used in the thesis. Note that this can take a while (Multiple weeks on a single Nvidia GTX 1080 Ti) and will require about 500 GB of free disc space.

## TODOs

- Add LICENSE
- Fill code documnetation holes
