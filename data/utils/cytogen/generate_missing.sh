#!/bin/bash

OUTPUT_DIR=$1

MISSING_PHANTOMS="164 212 290 497 689 825 869"
MISSING_FINAL="164 212 290 497 689 825 869 994 995 996 997 998 999 1000"

# Data directories 
PHANTOM_DIR="${OUTPUT_DIR}phantoms/" 
LABEL_DIR="${OUTPUT_DIR}labels/"
PLUGIN_DIR="/home/user/cytogen-plugins"

# Configs
INI_CONFIG_LOW_NOISE="hl60_low-noise.ini"
INI_CONFIG_MEDIUM_NOISE="hl60_medium-noise.ini"
INI_CONFIG_HIGH_NOISE="hl60_high-noise.ini"

# Phantom image generation
for i in $MISSING_PHANTOMS
do
    id=$(printf "%05d" $i)
    echo "Generating phantom ${i}..."
    3d-cytogen -c $INI_CONFIG_LOW_NOISE -p "${PHANTOM_DIR}phantom_${id}.tif" -l "${LABEL_DIR}label_${id}.tif" -d "$PLUGIN_DIR"
done

# Blurring images with optigen
for i in $MISSING_PHANTOMS
do
    id=$(printf "%05d" $i)
    echo "Generating blurred ${i}..."
    3d-optigen -c $INI_CONFIG_LOW_NOISE -p "${PHANTOM_DIR}phantom_${id}.tif" -b "${OUTPUT_DIR}blurred_low-noise/blurred_${id}.tif"
done

# Image acquisition
for i in $MISSING_PHANTOMS
do
    id=$(printf "%05d" $i)
    echo "Generating final low noise ${i}..."
    3d-acquigen -c $INI_CONFIG_LOW_NOISE -b "${OUTPUT_DIR}blurred_low-noise/blurred_${id}.tif" -f "${OUTPUT_DIR}images_low-noise/image_${id}.tif"
done

for i in $MISSING_FINAL
do
    id=$(printf "%05d" $i)
    echo "Generating final medium noise ${i}..."
    3d-acquigen -c $INI_CONFIG_MEDIUM_NOISE -b "${OUTPUT_DIR}blurred_low-noise/blurred_${id}.tif" -f "${OUTPUT_DIR}images_medium-noise/image_${id}.tif"
done

for i in $MISSING_FINAL
do
    id=$(printf "%05d" $i)
    echo "Generating final high noise ${i}..."
    3d-acquigen -c $INI_CONFIG_HIGH_NOISE -b "${OUTPUT_DIR}blurred_low-noise/blurred_${id}.tif" -f "${OUTPUT_DIR}images_high-noise/image_${id}.tif"
done