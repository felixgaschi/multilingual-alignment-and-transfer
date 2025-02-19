#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
MODEL=$3
STRATEGY=$4
SEED=$5
ADD_ARGS=$6

langs="ar bg de el es fr hi ru th tr vi zh"

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
#31,42,66,23,17
python scripts/2023_acl/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies ${STRATEGY}_dico \
    --models $MODEL \
    --tasks xnli \
    --cache_dir $CACHE_DIR \
    --n_epochs 3 \
    --seed $SEED \
    --right_langs $langs \
    --project_name "bylayer_" \
    --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}_realignment_seed_${SEED}.csv \
    --checkpoint_path $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}_realignment $ADD_ARGS