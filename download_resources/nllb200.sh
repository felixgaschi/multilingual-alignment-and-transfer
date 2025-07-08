#!/bin/bash

set -e


OUTPUT_DIR=$1

langs=$2

allenai_langs="amh bam ewe fon hau ibo kin lin lug luo mos nya gaz sna swh tsn twi wol xho yor zul"
opus_langs="am bm fon ha ig rw ln lg luo mos ny om sn st sw tn tw wo xh yo zu"

for lang in $langs; do

    # pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")
    pair=$(python -c "print('-'.join(sorted(['eng_Latn', '$lang'])))")

    if [ ! -d $OUTPUT_DIR/$pair ]; then
        # Download from OPUS - missing ee subset
        # wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$pair.txt.zip -O $OUTPUT_DIR/$pair.txt.zip
        # unzip -o $OUTPUT_DIR/$pair.txt.zip -d $OUTPUT_DIR/$pair
        # rm $OUTPUT_DIR/$pair.txt.zip

        wget https://storage.googleapis.com/allennlp-data-bucket/nllb/$pair.gz -O $OUTPUT_DIR/$pair.gz

        mkdir -p $OUTPUT_DIR/$pair/
        gzip -d < $OUTPUT_DIR/$pair.gz > $OUTPUT_DIR/$pair/$pair.tsv

        lines=$(wc -l < $OUTPUT_DIR/$pair/$pair.tsv)
        echo "$pair.tsv contain $lines sentences"

        awk -F'\t' '{print $1}' $OUTPUT_DIR/$pair/$pair.tsv > $OUTPUT_DIR/$pair/NLLB.$pair.eng_Latn
        awk -F'\t' '{print $2}' $OUTPUT_DIR/$pair/$pair.tsv > $OUTPUT_DIR/$pair/NLLB.$pair.$lang
        
        rm $OUTPUT_DIR/$pair.gz
    fi
done