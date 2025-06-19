#!/bin/bash

set -e

DATA_DIR=$1
DATASET=$2
STRATEGY=$3
SEED=$4
SELECTION_STRAT=$5
TASK=$6
ADD_ARGS=$7

#********************************************REALIGNMENT LANGUAGE SETTING********************************************
if [ "$SELECTION_STRAT" == "random_28" ]; then
    langs="ar tr hi lt el fr da fi zh th vi ja ko ta cs lv af da fi hu it ja ko lt no ro sk ta"

#=================================RANDOM DISTINCT FAMILY=================================
elif [ "$SELECTION_STRAT" == "random_distinct_family_14" ]; then
    langs="ar hi tr el fr th vi zh da fi ja ko lt ta"

elif [ "$SELECTION_STRAT" == "random_distinct_family_7" ]; then
    langs="ar hi tr el fr da lt"

elif [ "$SELECTION_STRAT" == "random_distinct_family_3" ]; then
    langs="ar hi tr"

#=================================INDO-EUROPEAN=================================
elif [ "$SELECTION_STRAT" == "indo_14" ]; then
    langs="de fr es it pt ro da sv el ru pl cs fa hi"

elif [ "$SELECTION_STRAT" == "indo_7" ]; then
    langs="de fr es it pt ro da"

elif [ "$SELECTION_STRAT" == "indo_3" ]; then
    langs="de fr es"

#=================================DIVERSE URIEL FEATURAL=================================
elif [ "$SELECTION_STRAT" == "DIVERSE_URIEL_FEATURAL_14" ]; then
    langs="ar el ru th tr vi zh af fa he ja ko sv ta"

elif [ "$SELECTION_STRAT" == "DIVERSE_URIEL_FEATURAL_7" ]; then
    langs="ar vi zh af he ja ta"

elif [ "$SELECTION_STRAT" == "DIVERSE_URIEL_FEATURAL_3" ]; then
    langs="ar zh af"

#=================================BASELINE=================================
elif [ "$SELECTION_STRAT" == "12_langs" ]; then
    langs="ar bg de el es fr hi ru th tr vi zh"

elif [ "$SELECTION_STRAT" == "34_langs" ]; then
    langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"

#=================================RESOURCE=================================
elif [ "$SELECTION_STRAT" == "high_res" ]; then
    langs="es de ja fr ar zh pt it ru hi sv pl ko tr"

elif [ "$SELECTION_STRAT" == "mix_res" ]; then
    langs="es de ja fr ar zh pt no bg da lt th he sk" # First half of each set

elif [ "$SELECTION_STRAT" == "low_res" ]; then
    langs="no bg da lt th he sk ro uk ta el lv sl af"

else
    langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"
    if [ "$SELECTION_STRAT" == "random_lang_with_seed" ]; then
        echo "Random langs 3 7 14."
        ADD_ARGS="--n_realignment_langs 3 7 14"
    else
        echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Setting to 34 langs."
    fi
fi
#********************************************END REALIGNMENT LANGUAGE SETTING********************************************

#********************************************TASK SETTING********************************************
if [ "$TASK" == "xnli" ]; then
    n_epochs=2
    eval_langs="ar bg de el es fr hi ru th tr vi zh amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul"
elif [ "$TASK" == "udpos" ]; then
    n_epochs=5
    eval_langs="bg cs de es lv af ar ca da el fa fi fr he hi hu it ja ko lt no pl pt ro ru sk sl sv ta th tr uk vi zh"
fi
#********************************************END TASK SETTING********************************************

# Print for confirmation
echo "Selected strategy: $SELECTION_STRAT"
echo "task: $TASK"
echo "langs: $langs"
echo "seeds: $SEED"
echo "epoch: $n_epochs"
echo "eval_langs: $eval_langs"

mkdir -p $DATA_DIR

CACHE_DIR=/home/bumie304/scratch/nlp_project/cache/
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=$DATA_DIR/phuoc_seed_31/$SELECTION_STRAT

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
for MODEL in "xlm-roberta-base" ; do
    python scripts/2025_alignfreeze_continuation/controlled_realignment.py \
        --translation_dir $TRANSLATION_DIR/$DATASET \
        --fastalign_dir $FASTALIGN_DIR/$DATASET \
        --dico_dir $DICOALIGN_DIR/$DATASET \
        --awesome_dir $AWESOME_DIR/$DATASET \
        --strategies $STRATEGY \
        --models $MODEL \
        --tasks $TASK \
        --cache_dir $CACHE_DIR \
        --n_epochs $n_epochs \
        --seed $SEED \
        --right_langs $langs \
        --eval_langs $eval_langs \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}__${TASK}.csv $ADD_ARGS \
        --checkpoint_path ~/scratch/nlp_project/results/$DATASET/$STRATEGY/$SELECTION_STRAT \
        --large_gpu $ADD_ARGS
done