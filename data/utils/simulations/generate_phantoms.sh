#!/bin/bash

NUM=$1
INI_CONFIG=$2
PHANTOM_DIR=$3
LABEL_DIR=$4
PLUGIN_DIR="/home/user/cytogen-plugins"

# Check output directories
if [ -d "$PHANTOM_DIR" ]; then
    echo "Phantom directory already exists. Aborting."
    exit 1
fi
if [ -d "$LABEL_DIR" ]; then
    echo "Label directory already exists. Aborting."
    exit 1
fi

mkdir "$PHANTOM_DIR"
mkdir "$LABEL_DIR"

# Phantom image generation
echo
echo
echo "GENERATING PHANTOM IMAGES"
echo "Start time $(date)"
echo
for i in $(seq 1 ${NUM})
do
    id=$(printf "%05d" $i)
    echo "Generating phantom ${i}..."
    3d-cytogen -c $INI_CONFIG -p "${PHANTOM_DIR}phantom_${id}.tif" -l "${LABEL_DIR}label_${id}.tif" -d "$PLUGIN_DIR"
done
