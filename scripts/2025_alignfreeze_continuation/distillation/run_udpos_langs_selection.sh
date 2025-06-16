#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
SELECTION_STRAT=$3
STRATEGY=$4
SEED=$5
ADD_ARGS=$6

# langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"

if [ "$SELECTION_STRAT" == "random_28" ]; then
    langs="ar tr hi lt el fr da fi zh th vi ja ko ta cs lv af da fi hu it ja ko lt no ro sk ta"

elif [ "$SELECTION_STRAT" == "random_14" ]; then
    langs="ar hi tr el fr th vi zh da fi ja ko lt ta"

elif [ "$SELECTION_STRAT" == "random_7" ]; then
    langs="ar hi tr el fr da lt"

elif [ "$SELECTION_STRAT" == "random_3" ]; then
    langs="ar hi tr"

else
    echo "Error: Unknown SELECTION_STRAT value: $SELECTION_STRAT"
    exit 1
fi

# Print for confirmation
echo "Selected strategy: $SELECTION_STRAT"
echo "langs: $langs"

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
#"freeze_high_anisotropy_dico"
# "distilbert-base-multilingual-cased" "bert-base-multilingual-cased"
for MODEL in "xlm-roberta-base"; do
    python scripts/2023_acl/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/$DATASET \
        --fastalign_dir $FASTALIGN_DIR/$DATASET \
        --dico_dir $DICOALIGN_DIR/$DATASET \
        --awesome_dir $AWESOME_DIR/$DATASET \
        --strategies before_dico \
        --models $MODEL \
        --tasks udpos \
        --cache_dir $CACHE_DIR \
        --n_epochs 5 \
        --seed $SEED \
        --right_langs $langs \
        --eval_langs bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}__udpos.csv $ADD_ARGS
done

