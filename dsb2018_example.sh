#!/bin/bash

EPOCHS=1000
BACKBONE=configs/backbones/resnet_unet.yaml
HEAD=configs/heads/stardist_default.yaml
TRAINING=configs/training/bs-8_early-stopping_reduce-lr.yaml
DATA=configs/data/stardist-dsb2018_F_L.yaml
INPUT_SIZE='[null, null, 1]'

python train.py -e $EPOCHS -b $BACKBONE --head $HEAD -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n resnet-unet2_dsb2018_stardist
