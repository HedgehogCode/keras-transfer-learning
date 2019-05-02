#!/bin/bash

NUM=$1
INI_CONFIG=$2
PHANTOM_DIR=$3
LABEL_DIR=$4
PLUGIN_DIR="/home/user/cytogen-plugins"

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
    phantom_file="${PHANTOM_DIR}phantom_${id}.tif"
    label_file="${LABEL_DIR}label_${id}.tif"
    if [ ! -f "${phantom_file}" ] && [ ! -f "${label_file}" ]; then
        echo "Generating phantom ${i}..."
        3d-cytogen -c $INI_CONFIG -p "${phantom_file}" -l "${label_file}" -d "$PLUGIN_DIR"
    elif [ ! -f "${phantom_file}" ] || [ ! -f "${label_file}" ]; then
        echo "ERROR: Phantom ${i} or label exists. This should not happen."
    else
        echo "Skipping. Phantom ${i} and label already present."
    fi
done
