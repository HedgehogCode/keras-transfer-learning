#!/bin/bash

NUM=1000
INI_CONFIG="granulocyte.ini"
OUTPUT_DIR=$1

./generate_dataset.sh $NUM $INI_CONFIG $OUTPUT_DIR "granulocyte"
