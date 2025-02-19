#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
MODEL=$3
ADD_ARGS=$4

#langs="ar es fr ru zh af fa hi"
langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"
langs="ar zh"

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/raw_results

mkdir -p $CACHE_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR

export DATA_DIR=$DATA_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR

if [ "$MODEL" = "distilbert-base-multilingual-cased" ] ; then
    MAX_LAYER=6
    echo "$MODEL"
elif [ "$MODEL" = "bert-base-multilingual-cased" ] ; then
    MAX_LAYER=12
    echo "$MODEL"
elif [ "$MODEL" = "xlm-roberta-base" ] ; then
    MAX_LAYER=12
    echo "$MODEL"
else
    exit 1
fi

python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies before_gradual_random_3_dico \
    --models $MODEL \
    --tasks xnli \
    --cache_dir $CACHE_DIR \
    --n_epochs 3 \
    --right_langs $langs \
    --project_prefix "bylayer_" \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__gradual_random_3_layer_realignment.csv $ADD_ARGS
