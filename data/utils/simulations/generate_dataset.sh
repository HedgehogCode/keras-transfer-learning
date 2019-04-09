#!/bin/bash

NUM=$1
INI_CONFIG=$2
OUTPUT_DIR=$3
OUTPUT_SUFFIX=$4

# Data directories
PHANTOM_DIR="${OUTPUT_DIR}phantoms/"
LABEL_DIR="${OUTPUT_DIR}labels/"
BLURRED_DIR="${OUTPUT_DIR}blurred_${OUTPUT_SUFFIX}/"
IMAGE_DIR="${OUTPUT_DIR}images_${OUTPUT_SUFFIX}/"

mkdir "$OUTPUT_DIR"

# Phantom image generation
./generate_phantoms.sh $NUM $INI_CONFIG $PHANTOM_DIR $LABEL_DIR

# Blurring images with optigen
./generate_blurred.sh $NUM $INI_CONFIG $PHANTOM_DIR $BLURRED_DIR

# Image acquisition
./generate_final.sh $NUM $INI_CONFIG $BLURRED_DIR $IMAGE_DIR
