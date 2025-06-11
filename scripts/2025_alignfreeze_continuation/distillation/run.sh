#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
STRATEGY=$3
SEED=$4
ADD_ARGS=$5

langs="ar bg de el es fr hi ru th tr vi zh"
additional_langs="cs lv af ca da fa fi he hu it ja ko lt no pl pt ro sk sl sv ta uk"


# Print for confirmation
echo "langs: $langs"
echo "additional_langs: $additional_langs"

mkdir -p $DATA_DIR

CACHE_DIR=/home/leelab-alignfreeze2/nlp_project/cache
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/results/$STRATEGY

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
export RESULT_DIR=$RESULT_DIR
#31,42,66,23,17
#"freeze_high_anisotropy_dico"
# "distilbert-base-multilingual-cased" "bert-base-multilingual-cased"
for MODEL in "xlm-roberta-base"; do
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
        --seed $SEED \
        --right_langs $langs \
        --eval_langs ar bg de el es fr hi ru th tr vi zh \
        --project_name "reg_lang_selection" \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}.csv \
        --additional_realignment_langs $additional_langs 
done