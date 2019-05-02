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
    blurred_file="${BLURRED_DIR}blurred_${id}.tif"
    final_file="${IMAGE_DIR}image_${id}.tif"
    if [ -f "${blurred_file}" ] && [ ! -f "${final_file}" ]; then
        echo "Generating final ${i}..."
        3d-acquigen -c $INI_CONFIG -b "${blurred_file}" -f "${blurred_file}"
    elif [ ! -f "${blurred_file}" ]; then
        echo "WARNING: Blurred ${i} missing."
    else
        echo "Skipping. Final ${i} already presnet."
    fi
done
