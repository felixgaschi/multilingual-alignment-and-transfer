#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
# SEED=$3
STRATEGY=$3
ADD_ARGS=$4

langs="ar bg de el es fr hi ru th tr vi zh"
additional_langs="cs lv af ca da fa fi he hu it ja ko lt no pl pt ro sk sl sv ta uk"

mkdir -p $DATA_DIR

CACHE_DIR=/home/bumie304/scratch/nlp_project/cache/
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/results/$STRATEGY
CHECKPOINT_DIR=/home/bumie304/scratch/nlp_project/results/$STRATEGY

mkdir -p $CACHE_DIR
mkdir -p $TRANSLATION_DIR
mkdir -p $FASTALIGN_DIR
mkdir -p $DICOALIGN_DIR
mkdir -p $AWESOME_DIR
mkdir -p $RESULT_DIR
mkdir -p $CHECKPOINT_DIR

export DATA_DIR=$DATA_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR
export FASTALIGN_DIR=$FASTALIGN_DIR
export DICOALIGN_DIR=$DICOALIGN_DIR
export AWESOME_DIR=$AWESOME_DIR
export RESULT_DIR=$RESULT_DIR
export CHECKPOINT_DIR=$CHECKPOINT_DIR
#31,42,66,23,17
#"freeze_high_anisotropy_dico"

for MODEL in "bert-base-multilingual-cased" "xlm-roberta-base" "distilbert-base-multilingual-cased"; do
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/$DATASET \
        --fastalign_dir $FASTALIGN_DIR/$DATASET \
        --dico_dir $DICOALIGN_DIR/$DATASET \
        --awesome_dir $AWESOME_DIR/$DATASET \
        --strategies $STRATEGY \
        --models $MODEL \
        --tasks xnli \
        --cache_dir $CACHE_DIR \
        --n_epochs 2 \
        --seed 31 42 66 \
        --right_langs $langs \
        --project_name "anisotropy" \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}.csv \
        --additional_realignment_langs $additional_langs \
        --checkpoint_path $CHECKPOINT_DIR/${MODEL}__${DATASET} $ADD_ARGS
done