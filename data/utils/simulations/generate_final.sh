#!/bin/bash

NUM=$1
INI_CONFIG=$2
BLURRED_DIR=$3
IMAGE_DIR=$4

# Check output directory
if [ -d "$IMAGE_DIR" ]; then
    echo "Output directory already exists. Aborting."
    exit 1
fi

mkdir "$IMAGE_DIR"

# Image acquisition
echo
echo
echo "GENERATING FINAL IMAGES"
echo "Start time $(date)"
echo
for i in $(seq 1 ${NUM})
do
    id=$(printf "%05d" $i)
    echo "Generating final ${i}..."
    3d-acquigen -c $INI_CONFIG -b "${BLURRED_DIR}blurred_${id}.tif" -f "${IMAGE_DIR}image_${id}.tif"
done
