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
elif [ "$SELECTION_STRAT" == "xt_only" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh"
elif [ "$SELECTION_STRAT" == "afri_only" ]; then
    langs="amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"

#=================================URIEL_DIVERSITY=================================
#=======================5_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_5" ]; then
    langs="fon_Latn ka kaz_Cyrl lin_Latn gaz_Latn"
elif [ "$SELECTION_STRAT" == "least_uriel_en_5" ]; then
    langs="it pt ro es uk"
elif [ "$SELECTION_STRAT" == "most_uriel_5" ]; then
    langs="ar lin_Latn gaz_Latn vi zh"
elif [ "$SELECTION_STRAT" == "least_uriel_5" ]; then
    langs="bn gu hi pa ur"

#=======================10_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_10" ]; then
    langs="af ar fon_Latn ka ja kaz_Cyrl lin_Latn gaz_Latn sna_Latn vi"
elif [ "$SELECTION_STRAT" == "least_uriel_en_10" ]; then
    langs="bg de el es fr it nl pt ro uk"
elif [ "$SELECTION_STRAT" == "most_uriel_10" ]; then
    langs="af ar zh ka kaz_Cyrl lin_Latn gaz_Latn sna_Latn ta vi"
elif [ "$SELECTION_STRAT" == "least_uriel_10" ]; then
    langs="bg el it lt pl pt ro ru es uk"

#=======================20_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_20" ]; then
    langs="af ar az eu zh fon_Latn lug_Latn ka el he ja kaz_Cyrl ko lin_Latn gaz_Latn sna_Latn ta twi_Latn vi yor_Latn"
elif [ "$SELECTION_STRAT" == "least_uriel_en_20" ]; then
    langs="bg nl et fi fr de el gu hi hu it lt fa pl pt pa ro ru es uk"
#Similar to with en
# elif [ "$SELECTION_STRAT" == "most_uriel_20" ]; then
#     langs="af ar az eu zh fon_Latn lug_Latn ka el he ja kaz_Cyrl ko lin_Latn gaz_Latn sna_Latn ta twi_Latn vi yor_Latn"
# elif [ "$SELECTION_STRAT" == "least_uriel_20" ]; then
#     langs="bn gu hi pa ur"

#=======================40_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_40" ]; then
    langs="af ar az eu mya_Mymr zh ewe_Latn fon_Latn fr lug_Latn ka el hau_Latn he ibo_Latn ja kaz_Cyrl kin_Latn ko lin_Latn ms ml mar_Deva nya_Latn gaz_Latn fa ru sna_Latn es tgl_Latn ta tel_Telu th tr twi_Latn ur vi xho_Latn yor_Latn zul_Latn"
elif [ "$SELECTION_STRAT" == "least_uriel_en_40" ]; then
    langs="amh_Ethi ar az bam_Latn eu bn bg nl et fi fr de el gu hau_Latn he hi hu id it jav_Latn lt luo_Latn ml mar_Deva mos_Latn fa pl pt pa ro ru es tgl_Latn ta tel_Telu tr uk ur wol_Latn"
# elif [ "$SELECTION_STRAT" == "least_uriel_40" ]; then
#     langs="bn gu hi pa ur"
# elif [ "$SELECTION_STRAT" == "most_uriel_40" ]; then
#     langs="ar lin_Latn gaz_Latn vi zh"

#=================================FAMILY_DIVERSITY=================================
# MOST DIVERSE DISTINCT FAMILY: add English to the set, so need to "disperse" from English
# LEAST DIVERSE DISTINCT FAMILY: Indo European only
#=======================DISTINCT_FAMILY_5
elif [ "$SELECTION_STRAT" == "most_family_en_5" ]; then
    langs="ka kaz_Cyrl lin_Latn gaz_Latn vi"
elif [ "$SELECTION_STRAT" == "least_family_en_5" ]; then
    langs="af nl de it pt"

#=======================DISTINCT_FAMILY_10
elif [ "$SELECTION_STRAT" == "most_family_en_10" ]; then
    langs="ar zh ka ja kaz_Cyrl lin_Latn ms gaz_Latn ta vi"
elif [ "$SELECTION_STRAT" == "least_family_en_5" ]; then
    langs="af bg nl fr de it pt ro es uk"

#=======================DISTINCT_FAMILY_20
elif [ "$SELECTION_STRAT" == "most_family_en_20" ]; then
    langs="ar az eu zh fr ka el hau_Latn ja kaz_Cyrl ko lin_Latn luo_Latn ms mar_Deva gaz_Latn ru ta th vi"
elif [ "$SELECTION_STRAT" == "least_family_en_20" ]; then
    langs="af bn bg nl fr de el gu hi it lt mar_Deva pl pt pa ro ru es uk ur"

#=======================DISTINCT_FAMILY_25
elif [ "$SELECTION_STRAT" == "most_family_en_25" ]; then
    langs="ar az bam_Latn eu zh fi fr ka el hau_Latn hu ja kaz_Cyrl ko lin_Latn luo_Latn ms mar_Deva mos_Latn gaz_Latn ru ta tel_Telu th vi"

#=================================RESOURCE_LEVEL=================================
#=======================HRLS

#=======================MRLS

#=======================LRLS


#=================================RANDOM=================================
else
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
    if [ "$SELECTION_STRAT" == "random_langs_with_seed_5" ]; then
        echo "Random langs 5."
        ADD_ARGS="--n_realignment_langs 5"
    elif [ "$SELECTION_STRAT" == "random_langs_with_seed_10" ]; then
        echo "Random langs 10."
        ADD_ARGS="--n_realignment_langs 10"
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
    eval_langs="ar bg de el es fr hi ru sw th tr ur vi zh amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul"
elif [ "$TASK" == "wikiann" ]; then
    n_epochs=5
    eval_langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
elif [ "$TASK" == "udpos" ]; then
    n_epochs=5
    eval_langs="af ar bg de el es et eu fa fi fr he hi hu id it ja kk ko lt mr nl pl pt ro ru ta te th tl tr uk ur vi wo yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
elif [ "$TASK" == "xquad" ]; then
    n_epochs=5
    eval_langs="ar de el es hi ru th tr vi zh id"
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
        --realignment_steps 24544 \
        --right_langs $langs \
        --eval_langs $eval_langs \
        --output_file $RESULT_DIR/${MODEL}__${DATASET}__${STRATEGY}__${TASK}.csv $ADD_ARGS \
        --checkpoint_path ~/scratch/nlp_project/results/$DATASET/$STRATEGY/$SELECTION_STRAT \
        --large_gpu $ADD_ARGS
done