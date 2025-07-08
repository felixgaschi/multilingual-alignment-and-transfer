#!/bin/bash

set -e


OUTPUT_DIR=$1

wget https://tinyurl.com/flores200dataset -O $OUTPUT_DIR/flores200_dataset.tar.gz

tar -xvzf $OUTPUT_DIR/flores200_dataset.tar.gz -C $OUTPUT_DIR

dev_dir=$OUTPUT_DIR/flores200_dataset/dev
devtest_dir=$OUTPUT_DIR/flores200_dataset/devtest

# Loop through all .dev files
for dev_file in $dev_dir/*.dev; do
    # Extract filename without path and extension
    filename=$(basename "$dev_file" .dev)
    
    devtest_file=$devtest_dir/$filename.devtest
    output_file=$OUTPUT_DIR/$filename.all

    # Check if corresponding .devtest exists
    if [[ -f $devtest_file ]]; then
        cat $dev_file $devtest_file > $output_file
        echo "Created $output_file"
    else
        echo "Warning: Missing devtest file for $filename"
    fi
done  

rm $OUTPUT_DIR/flores200_dataset.tar.gz
rm -rf $OUTPUT_DIR/flores200_dataset