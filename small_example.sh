#!/bin/bash

EPOCHS=10
BACKBONE=configs/backbones/unet-very-small.yaml
BACKBONE_PRETRAINED=configs/backbones/unet-very-small_pretrained.yaml
TRAINING=configs/training/bs-8_early-stopping_reduce-lr.yaml
DATA=configs/data/stardist-dsb2018_small-patches_basic-dataaug.yaml
INPUT_SIZE='[null, null, 1]'

HEAD_UNET=configs/heads/fgbg-segm_default.yaml
HEAD_STARDIST=configs/heads/stardist_default.yaml

python train.py -e $EPOCHS -b $BACKBONE --head $HEAD_UNET -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n small-unet_stardist-dsb2018
python train.py -e $EPOCHS -b $BACKBONE --head $HEAD_STARDIST -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n small-stardist_stardist-dsb2018
python train.py -e $EPOCHS -b $BACKBONE_PRETRAINED --head $HEAD_STARDIST -t $TRAINING -d $DATA -i "$INPUT_SIZE" -n small-stardist_stardist-dsb2018_pretrained