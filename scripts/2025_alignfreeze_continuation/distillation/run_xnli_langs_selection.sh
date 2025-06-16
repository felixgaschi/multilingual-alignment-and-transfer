#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
STRATEGY=$3
SEED=$4
SELECTION_STRAT=$5
ADD_ARGS=$6


if [ "$SELECTION_STRAT" == "random_28" ]; then
    langs="ar tr hi lt el fr da fi zh th vi ja ko ta"
    additional_langs="cs lv af da fi hu it ja ko lt no ro sk ta"

elif [ "$SELECTION_STRAT" == "random_14" ]; then
    langs="ar hi tr el fr th vi zh"
    additional_langs="da fi ja ko lt ta"

elif [ "$SELECTION_STRAT" == "random_7" ]; then
    langs="ar hi tr el fr"
    additional_langs="da lt"

elif [ "$SELECTION_STRAT" == "random_3" ]; then
    langs="ar hi tr"
    additional_langs=""

else
    echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Setting to default."
    langs="ar bg de el es fr hi ru th tr vi zh"
    additional_langs="cs lv af ca da fa fi he hu it ja ko lt no pl pt ro sk sl sv ta uk"
fi

# Print for confirmation
echo "Selected strategy: $SELECTION_STRAT"
echo "langs: $langs"
echo "additional_langs: $additional_langs"

mkdir -p $DATA_DIR

CACHE_DIR=/home/bumie304/scratch/nlp_project/cache/
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/reg_lang_selection_results/$SELECTION_STRAT

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
        --eval_langs ar bg de el es fr hi ru th tr vi zh afrixnli\
        --project_name "reg_lang_selection" \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}.csv \
        --additional_realignment_langs $additional_langs 
done