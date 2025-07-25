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
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jav_Latn ka kaz_Cyrl ko lt mar_Deva ml ms mya_Mymr nl pa pl pt ro ru swh_Latn ta tel_Telu tgl_Latn th tr uk ur vi wol_Latn yor_Latn zh"
elif [ "$SELECTION_STRAT" == "afri_only" ]; then
    langs="amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn"

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

#=======================40_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_40" ]; then
    langs="af ar az eu mya_Mymr zh ewe_Latn fon_Latn fr lug_Latn ka el hau_Latn he ibo_Latn ja kaz_Cyrl kin_Latn ko lin_Latn ms ml mar_Deva nya_Latn gaz_Latn fa ru sna_Latn es tgl_Latn ta tel_Telu th tr twi_Latn ur vi xho_Latn yor_Latn zul_Latn"
elif [ "$SELECTION_STRAT" == "least_uriel_en_40" ]; then
    langs="amh_Ethi ar az bam_Latn eu bn bg nl et fi fr de el gu hau_Latn he hi hu id it jav_Latn lt luo_Latn ml mar_Deva mos_Latn fa pl pt pa ro ru es tgl_Latn ta tel_Telu tr uk ur wol_Latn"

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
elif [ "$SELECTION_STRAT" == "least_family_en_10" ]; then
    langs="af bg nl fr de it pt ro es uk"

#=======================DISTINCT_FAMILY_20
elif [ "$SELECTION_STRAT" == "most_family_en_20" ]; then
    langs="ar az eu zh fr ka el hau_Latn ja kaz_Cyrl ko lin_Latn luo_Latn ms mar_Deva gaz_Latn ru ta th vi"
elif [ "$SELECTION_STRAT" == "least_family_en_20" ]; then
    langs="af bn bg nl fr de el gu hi it lt mar_Deva pl pt pa ro ru es uk ur"

#=======================DISTINCT_FAMILY_25
elif [ "$SELECTION_STRAT" == "most_family_en_25" ]; then
    langs="ar az bam_Latn eu zh fi fr ka el hau_Latn hu ja kaz_Cyrl ko lin_Latn luo_Latn ms mar_Deva mos_Latn gaz_Latn ru ta tel_Telu th vi"

#=================================SCRIPT_DIVERSITY=================================
#=======================MOST_DIVERSE_DISTINCT_SCRIPT
elif [ "$SELECTION_STRAT" == "most_distinct_script_5" ]; then
    langs="ar ka ja kaz_Cyrl th"
elif [ "$SELECTION_STRAT" == "most_distinct_script_10" ]; then
    langs="ar mya_Mymr zh ka el he ja kaz_Cyrl ta th"
elif [ "$SELECTION_STRAT" == "most_distinct_script_18" ]; then
    langs="amh_Ethi ar bn mya_Mymr zh ka el gu he hi ja kaz_Cyrl ko ml pa ta tel_Telu th"

#=======================MOST_DIVERSE_SAME_SCRIPT (Latin only)
elif [ "$SELECTION_STRAT" == "most_same_script_5" ]; then
    langs="az fon_Latn lin_Latn gaz_Latn tgl_Latn"
elif [ "$SELECTION_STRAT" == "most_same_script_10" ]; then
    langs="af az eu fon_Latn lin_Latn gaz_Latn sna_Latn tgl_Latn twi_Latn vi"
elif [ "$SELECTION_STRAT" == "most_same_script_20" ]; then
    langs="bam_Latn nl et fi fr de hau_Latn hu id it jav_Latn lt luo_Latn ms pl pt ro es tgl_Latn wol_Latn"
elif [ "$SELECTION_STRAT" == "most_same_script_41" ]; then
    langs="af az bam_Latn eu nl et ewe_Latn fi fon_Latn fr lug_Latn de hau_Latn hu ibo_Latn id it jav_Latn kin_Latn lin_Latn lt luo_Latn ms mos_Latn nya_Latn gaz_Latn pl pt ro sna_Latn es swh_Latn tgl_Latn tsn_Latn tr twi_Latn vi wol_Latn xho_Latn yor_Latn zul_Latn"

#=======================LEAST_DIVERSE_SAME_SCRIPT (Latin only)
elif [ "$SELECTION_STRAT" == "least_same_script_5" ]; then
    langs="nl fr de it pt"
elif [ "$SELECTION_STRAT" == "least_same_script_10" ]; then
    langs="nl et fi fr de hu it pt ro es"
elif [ "$SELECTION_STRAT" == "least_same_script_20" ]; then
    langs="af az eu ewe_Latn fon_Latn fr lug_Latn lin_Latn lt ms gaz_Latn pl sna_Latn es tgl_Latn tr twi_Latn vi yor_Latn zul_Latn"

#=================================ABLATION_10=================================

#_________________________________JOSHI CLASS
#=======================Joshi 4 and 5
elif [ "$SELECTION_STRAT" == "abla_random_joshi45" ]; then
    langs="ar de es eu fa fi fr hi hu it ja ko nl pl pt ru tr vi zh"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_joshi45" ]; then
    langs="ar eu zh fr ja ko fa ru tr vi"
elif [ "$SELECTION_STRAT" == "abla_most_script_joshi45" ]; then
    langs="ar eu fr hi ja ko fa ru tr vi"

#=======================Joshi 3
elif [ "$SELECTION_STRAT" == "abla_random_joshi3" ]; then
    langs="af bg bn el et he id ka kaz_Cyrl lt ms ro ta tgl_Latn th uk ur"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_joshi3" ]; then
    langs="af ka el he kaz_Cyrl ms tgl_Latn ta th ur"
elif [ "$SELECTION_STRAT" == "abla_most_family_joshi3" ]; then
    langs="et ka el he kaz_Cyrl lt ms ta th ur"
elif [ "$SELECTION_STRAT" == "abla_most_script_joshi3" ]; then
    langs="bn bg ka el he kaz_Cyrl ta th uk ur"

#=======================Joshi 3-5
elif [ "$SELECTION_STRAT" == "abla_random_joshi35" ]; then
    langs="af ar bg bn de el es et eu fa fi fr he hi hu id it ja ka kaz_Cyrl ko lt ms nl pl pt ro ru ta tgl_Latn th tr uk ur vi zh"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_joshi35" ]; then
    langs="af ar eu zh ka ja kaz_Cyrl ms ta vi"
elif [ "$SELECTION_STRAT" == "abla_most_family_joshi35" ]; then
    langs="ar zh ka el ja kaz_Cyrl ms ta th vi"
elif [ "$SELECTION_STRAT" == "abla_most_script_joshi35" ]; then
    langs="ar zh ka el he ja kaz_Cyrl ko ta th"

#=======================Joshi < 2
elif [ "$SELECTION_STRAT" == "abla_random_joshi2" ]; then
    langs="amh_Ethi az bam_Latn ewe_Latn fon_Latn gaz_Latn gu hau_Latn ibo_Latn jav_Latn kin_Latn lin_Latn lug_Latn luo_Latn mar_Deva ml mos_Latn mya_Mymr nya_Latn pa sna_Latn swh_Latn tel_Telu tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_joshi2" ]; then
    langs="az mya_Mymr fon_Latn kin_Latn lin_Latn mar_Deva gaz_Latn sna_Latn tel_Telu yor_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_family_joshi2" ]; then
    langs="az mya_Mymr hau_Latn jav_Latn lin_Latn luo_Latn mar_Deva mos_Latn gaz_Latn tel_Telu"
elif [ "$SELECTION_STRAT" == "abla_most_script_joshi2" ]; then
    langs="amh_Ethi mya_Mymr gu lin_Latn ml mar_Deva gaz_Latn pa tel_Telu yor_Latn"

#_________________________________SEEN AND UNSEEN
#=======================Seen-only XLM-R
elif [ "$SELECTION_STRAT" == "abla_random_seen_xlmr" ]; then
    langs="af amh_Ethi ar az bg bn de el es et eu fa fi fr gaz_Latn gu hau_Latn he hi hu id it ja jav_Latn ka kaz_Cyrl ko lt mar_Deva ml ms mya_Mymr nl pa pl pt ro ru swh_Latn ta tel_Telu tgl_Latn th tr uk ur vi xho_Latn zh"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_seen_xlmr" ]; then
    langs="af ar zh ka ja kaz_Cyrl ms gaz_Latn swh_Latn vi"
elif [ "$SELECTION_STRAT" == "abla_most_family_seen_xlmr" ]; then
    langs="ar zh ka ja kaz_Cyrl ms gaz_Latn ta vi xho_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_script_seen_xlmr" ]; then
    langs="ar mya_Mymr zh ka el he ja kaz_Cyrl ta th"

#=======================Unseen-only XLM-R
elif [ "$SELECTION_STRAT" == "abla_random_unseen_xlmr" ]; then
    langs="bam_Latn ewe_Latn fon_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn sna_Latn tsn_Latn twi_Latn wol_Latn yor_Latn zul_Latn"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_unseen_xlmr" ]; then
    langs="ewe_Latn fon_Latn lin_Latn luo_Latn mos_Latn sna_Latn twi_Latn wol_Latn yor_Latn zul_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_family_unseen_xlmr" ]; then
    langs="bam_Latn fon_Latn lin_Latn luo_Latn mos_Latn sna_Latn twi_Latn wol_Latn yor_Latn zul_Latn"

#=======================Seen-only mBERT
elif [ "$SELECTION_STRAT" == "abla_random_seen_mbert" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jav_Latn ka kaz_Cyrl ko lt mar_Deva ml ms mya_Mymr nl pa pl pt ro ru swh_Latn ta tel_Telu tgl_Latn th tr uk ur vi yor_Latn zh"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_seen_mbert" ]; then
    langs="af ar zh ka ja kaz_Cyrl swh_Latn ta vi yor_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_family_seen_mbert" ]; then
    langs="ar az zh ka ja kaz_Cyrl ms ta vi yor_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_script_seen_mbert" ]; then
    langs="ar mya_Mymr zh ka el he ja kaz_Cyrl ta th"

#=======================Unseen-only mBERT
elif [ "$SELECTION_STRAT" == "abla_random_unseen_mbert" ]; then
    langs="amh_Ethi bam_Latn ewe_Latn fon_Latn gaz_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn sna_Latn tsn_Latn twi_Latn wol_Latn xho_Latn zul_Latn"
    ADD_ARGS="--n_realignment_langs 10"
elif [ "$SELECTION_STRAT" == "abla_most_uriel_unseen_mbert" ]; then
    langs="amh_Ethi ewe_Latn fon_Latn hau_Latn lin_Latn luo_Latn gaz_Latn sna_Latn twi_Latn xho_Latn"
elif [ "$SELECTION_STRAT" == "abla_most_family_unseen_mbert" ]; then
    langs="amh_Ethi bam_Latn fon_Latn hau_Latn lin_Latn luo_Latn mos_Latn gaz_Latn sna_Latn twi_Latn"


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
        echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Exitting."
        exit 1
    fi
fi
#********************************************END REALIGNMENT LANGUAGE SETTING********************************************

#********************************************TASK SETTING********************************************
# THERE ARE DUPLICATED LANGS IN EVAL SET!!!
if [ "$TASK" == "xnli" ]; then
    # 14 XTREME-R + 18 AfriXNLI + 10 AmericasNLI + Ind + Mya 
    n_epochs=2
    eval_langs="ar bg de el es fr hi ru sw th tr ur vi zh amh eng ewe fra hau ibo kin lin lug orm sna sot swa twi wol xho yor zul aym bzd cni gn hch nah oto quy shp tar ind mya"
elif [ "$TASK" == "wikiann" ]; then
    # 47 XTREME-R + 20 MasakhaNER2
    n_epochs=5
    eval_langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jv ka kk ko lt ml mr ms my nl pa pl pt qu ro ru sw ta te th tl tr uk ur vi yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
elif [ "$TASK" == "xtreme_r.udpos" ]; then
    # 37 XTREME-R + 20 MasakhaNER2
    n_epochs=5
    eval_langs="af ar bg de el es et eu fa fi fr he hi hu id it ja kk ko lt mr nl pl pt ro ru ta te th tl tr uk ur vi wo yo zh bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul"
elif [ "$TASK" == "xquad" ]; then
    # 10 XTREME-R + Ind
    n_epochs=5
    eval_langs="ar de el es hi ru th tr vi zh ind"
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