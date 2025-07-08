#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

for lang in $langs; do

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    if [ ! -d $OUTPUT_DIR/$pair ]; then

        wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$pair.txt.zip -O $OUTPUT_DIR/$pair.txt.zip

        unzip -o $OUTPUT_DIR/$pair.txt.zip -d $OUTPUT_DIR/$pair

        rm $OUTPUT_DIR/$pair.txt.zip
    fi
done