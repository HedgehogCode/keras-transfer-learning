#!/bin/bash

EPOCHS=100
BACKBONE=configs/backbones/unet-csbdeep.yaml
HEAD=configs/heads/stardist_default.yaml
TRAINING=configs/training/bs-8_early-stopping_reduce-lr.yaml
DATA=configs/data/stardist-dsb2018_F_L.yaml
INPUT_SIZE='[null, null, 1]'

python train.py -e $EPOCHS -b $BACKBONE --head $HEAD -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n stardist-dsb2018_F_L__stardist
