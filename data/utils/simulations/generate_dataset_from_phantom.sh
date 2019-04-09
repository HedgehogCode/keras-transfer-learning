#!/bin/bash

INI_CONFIG=$1
OUTPUT_DIR=$2
OUTPUT_SUFFIX=$3
PLUGIN_DIR="../out/plugins"

# Data directories
PHANTOM_DIR="${OUTPUT_DIR}phantoms/"
BLURRED_DIR="${OUTPUT_DIR}blurred_${OUTPUT_SUFFIX}/"
IMAGE_DIR="${OUTPUT_DIR}images_${OUTPUT_SUFFIX}/"

# Number of images
NUM=$(ls -l ${PHANTOM_DIR}*.tif | wc -l)

# Blurring images with optigen
./generate_blurred.sh $NUM $INI_CONFIG $PHANTOM_DIR $BLURRED_DIR

# Image acquisition
./generate_final.sh $NUM $INI_CONFIG $BLURRED_DIR $IMAGE_DIR

