#!/bin/bash

BACKBONE=configs/backbones/resnet-unet.yaml
HEAD=configs/heads/stardist.yaml
TRAINING=configs/training/bs-2_early-stopping_reduce-lr.yaml
EVALUATION=configs/eval/instance_segm.yaml

python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-0gamma.yaml -i "[null, null, 1]"      -n Z02_resnet-unet_stardist_dsb2018-aug-0gamma_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-1sharpen.yaml -i "[null, null, 1]"    -n Z02_resnet-unet_stardist_dsb2018-aug-1sharpen_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-2emboss.yaml -i "[null, null, 1]"     -n Z02_resnet-unet_stardist_dsb2018-aug-2emboss_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-3add.yaml -i "[null, null, 1]"        -n Z02_resnet-unet_stardist_dsb2018-aug-3add_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-4gaussnoise.yaml -i "[null, null, 1]" -n Z02_resnet-unet_stardist_dsb2018-aug-4gaussnoise_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-5gaussblur.yaml -i "[null, null, 1]"  -n Z02_resnet-unet_stardist_dsb2018-aug-5gaussblur_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-6motion.yaml -i "[null, null, 1]"     -n Z02_resnet-unet_stardist_dsb2018-aug-6motion_R_F
python -u train.py -e 1000 -b $BACKBONE --head $HEAD -t $TRAINING --eval $EVALUATION -d configs/data/dsb2018-aug-7invert.yaml -i "[null, null, 1]"     -n Z02_resnet-unet_stardist_dsb2018-aug-7invert_R_F

python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-0gamma_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-1sharpen_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-2emboss_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-3add_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-4gaussnoise_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-5gaussblur_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-6motion_R_F/config.yaml
python -u evaluate.py models/Z/02_resnet-unet_stardist_dsb2018-aug-7invert_R_F/config.yaml
