#!/bin/bash

EPOCHS=1000
BACKBONE=configs/backbones/unet-csbdeep.yaml
HEAD=configs/heads/segm_cityscapes.yaml
TRAINING=configs/training/bs-8_early-stopping_reduce-lr.yaml
DATA=configs/data/cityscapes_F_L.yaml
INPUT_SIZE='[null, null, 1]'

python train.py -e $EPOCHS -b $BACKBONE --head $HEAD -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n unet_cityscapes
