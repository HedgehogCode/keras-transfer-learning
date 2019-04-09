#!/bin/bash

NUM=1000
INI_CONFIG_LOW_NOISE="hl60_low-noise.ini"
INI_CONFIG_MEDIUM_NOISE="hl60_medium-noise.ini"
INI_CONFIG_HIGH_NOISE="hl60_high-noise.ini"
OUTPUT_DIR=$1

# Low noise
./generate_dataset.sh $NUM $INI_CONFIG_LOW_NOISE $OUTPUT_DIR "low-noise"

# Medium noise
./generate_dataset_from_blurred.sh $INI_CONFIG_MEDIUM_NOISE $OUTPUT_DIR "low-noise" "medium-noise"

# High noise
./generate_dataset_from_blurred.sh $INI_CONFIG_HIGH_NOISE $OUTPUT_DIR "low-noise" "high-noise"
