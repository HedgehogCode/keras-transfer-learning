#!/bin/bash

INI_CONFIG=$1
OUTPUT_DIR=$2
INPUT_SUFFIX=$3
OUTPUT_SUFFIX=$4
PLUGIN_DIR="../out/plugins"

# Data directories
BLURRED_DIR="${OUTPUT_DIR}blurred_${INPUT_SUFFIX}/"
IMAGE_DIR="${OUTPUT_DIR}images_${OUTPUT_SUFFIX}/"

# Number of images
NUM=$(ls -l ${BLURRED_DIR}*.tif | wc -l)

# Image acquisition
./generate_final.sh $NUM $INI_CONFIG $BLURRED_DIR $IMAGE_DIR

