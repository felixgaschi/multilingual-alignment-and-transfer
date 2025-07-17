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

#=================================BASELINE=================================
if [ "$SELECTION_STRAT" == "xt_afri" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"

else
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
    if [ "$SELECTION_STRAT" == "random_langs_with_seed_5" ]; then
        echo "Random langs 5."
        ADD_ARGS="--n_realignment_langs 5"
    elif [ "$SELECTION_STRAT" == "random_langs_with_seed_20" ]; then
        echo "Random langs 20."
        ADD_ARGS="--n_realignment_langs 20"
    elif [ "$SELECTION_STRAT" == "random_langs_with_seed_40" ]; then
        echo "Random langs 40."
        ADD_ARGS="--n_realignment_langs 40"
    else
        echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Setting to XTREME-R and African languages."
    fi
fi
#********************************************END REALIGNMENT LANGUAGE SETTING********************************************

#********************************************TASK SETTING********************************************
if [ "$TASK" == "xnli" ]; then
    n_epochs=2
    eval_langs="ar bg de el en es fr hi ru sw th tr ur vi zh amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul"
elif [ "$TASK" == "wikiann" ]; then
    n_epochs=5
    eval_langs="af ar az bg bn de el en es et eu fa fi fr gu he hi hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
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

CACHE_DIR=~/scratch/nlp_project/cache/
TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=scripts/2025_aacl/results/$SELECTION_STRAT

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
for MODEL in "xlm-roberta-base" "bert-base-multilingual-cased"; do
    uv run scripts/2025_aacl/controlled_realignment.py \
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