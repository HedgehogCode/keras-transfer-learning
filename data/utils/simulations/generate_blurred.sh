#!/bin/bash

NUM=$1
INI_CONFIG=$2
PHANTOM_DIR=$3
BLURRED_DIR=$4

# Check output directorie
if [ -d "$BLURRED_DIR" ]; then
    echo "Blurred output directory already exists. Aborting."
    exit 1
fi

mkdir "$BLURRED_DIR"

# Blurring images with optigen
echo
echo
echo "GENERATING BLURRED IMAGES"
echo "Start time $(date)"
echo
for i in $(seq 1 ${NUM})
do
    id=$(printf "%05d" $i)
    echo "Generating blurred ${i}..."
    3d-optigen -c $INI_CONFIG -p "${PHANTOM_DIR}phantom_${id}.tif" -b "${BLURRED_DIR}blurred_${id}.tif"
done
