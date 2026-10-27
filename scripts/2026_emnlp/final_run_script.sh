#!/bin/bash

set -e

STRATEGY=$1
SELECTION_STRAT=$2
MODEL=$3
TASK=$4
SEED=$5
MAX_REALIGNMENT_STEPS=$6
REALIGNMENT_BATCH_SIZE=$7
ADD_ARGS=$8

DATA_DIR=./data
CACHE_DIR=./cache
CKPT_DIR=./results/

DATASET="mix_opus100_nllb"

#********************************************REALIGNMENT LANGUAGE SETTING********************************************

#=================================BASELINE=================================
if [ "$SELECTION_STRAT" == "xt_afri" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
elif [ "$SELECTION_STRAT" == "xt_only" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja jav_Latn ka kaz_Cyrl ko lt mar_Deva ml ms mya_Mymr nl pa pl pt ro ru swh_Latn ta tel_Telu tgl_Latn th tr uk ur vi wol_Latn yor_Latn zh"
elif [ "$SELECTION_STRAT" == "afri_only" ]; then
    langs="amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn"

#=================================URIEL_DIVERSITY=================================
#=======================40_langs
elif [ "$SELECTION_STRAT" == "most_uriel_en_40" ]; then
    langs="af ar az eu mya_Mymr zh ewe_Latn fon_Latn fr lug_Latn ka el hau_Latn he ibo_Latn ja kaz_Cyrl kin_Latn ko lin_Latn ms ml mar_Deva nya_Latn gaz_Latn fa ru sna_Latn es tgl_Latn ta tel_Telu th tr twi_Latn ur vi xho_Latn yor_Latn zul_Latn"
elif [ "$SELECTION_STRAT" == "least_uriel_en_40" ]; then
    langs="amh_Ethi ar az bam_Latn eu bn bg nl et fi fr de el gu hau_Latn he hi hu id it jav_Latn lt luo_Latn ml mar_Deva mos_Latn fa pl pt pa ro ru es tgl_Latn ta tel_Telu tr uk ur wol_Latn"

#=================================RANDOM=================================
elif [ "$STRATEGY" == "baseline" ]; then
    langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
    echo "Fine-tuning baseline."
    SELECTION_STRAT="ft_only"
else
    echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Exitting."
    exit 1
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
elif [[ "$TASK" == *"clirmatrix"* ]]; then
    # Last 5 langs are not included in realignment
    n_epochs=0
    eval_langs="afr amh ar az bg bn de el es et eu fa fi fr gu hau he hi hu ibo id it ja jv ka kk ko lt ml mr ms my nl pa pl pt ro ru sna swa ta te th tl tr twi uk ur vi wol yor zh zul ary tir arz som nso"
elif [[ "$TASK" == "sib200" ]]; then
    # All realignment langs
    n_epochs=5
    eval_langs="afr_Latn amh_Ethi arb_Arab azj_Latn bam_Latn ben_Beng bul_Cyrl deu_Latn ell_Grek est_Latn eus_Latn ewe_Latn fin_Latn fon_Latn fra_Latn gaz_Latn guj_Gujr hau_Latn heb_Hebr hin_Deva hun_Latn ibo_Latn ind_Latn ita_Latn jav_Latn jpn_Jpan kat_Geor kaz_Cyrl kin_Latn kor_Hang lin_Latn lit_Latn lug_Latn luo_Latn mal_Mlym mar_Deva mos_Latn mya_Mymr nld_Latn nya_Latn pan_Guru pol_Latn por_Latn prs_Arab ron_Latn rus_Cyrl sna_Latn spa_Latn swh_Latn tam_Taml tel_Telu tgl_Latn tha_Thai tsn_Latn tur_Latn twi_Latn ukr_Cyrl urd_Arab vie_Latn wol_Latn xho_Latn yor_Latn zho_Hans zsm_Latn zul_Latn"
elif [[ "$TASK" == *"xlsum"* ]]; then
    if [[ "$MODEL" =~ [gG]emma|[lL]lama ]]; then
        n_epochs=1
    else
        n_epochs=5
    fi
    eval_langs="oromo french amharic arabic azerbaijani bengali burmese chinese_simplified chinese_traditional welsh english kirundi gujarati hausa hindi igbo indonesian japanese korean kyrgyz marathi spanish scottish_gaelic nepali pashto persian pidgin portuguese punjabi russian serbian_cyrillic serbian_latin sinhala somali swahili tamil telugu thai tigrinya turkish ukrainian urdu uzbek vietnamese yoruba"
    # eval_langs="spanish arabic bengali amharic scottish_gaelic"
fi
#********************************************END TASK SETTING********************************************

# Print for confirmation
echo "ADD_ARGS: $ADD_ARGS"
echo "model: $MODEL"
echo "task: $TASK"
echo "Max realignment steps: $MAX_REALIGNMENT_STEPS"
echo "Selected realignment set: $SELECTION_STRAT, realignment langs: $langs"
echo "seeds: $SEED"
echo "epoch: $n_epochs"
echo "eval_langs: $eval_langs"

mkdir -p $DATA_DIR

ADD_ARGS_SAVE_NAME="${ADD_ARGS//--learning_rate /__lr}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--enable_weighted_sampling/__weighted}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--weighted_sampling_method /__}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--ucb_exploration_coef /__excoef}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--inner_batches_before_outer /__innerb}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--meta_learning_rate /__metalr}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lbsmoothing_eps/__smeps}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--with_regularization/__withreg}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lambda_entropy /__lamdae}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--use_adapter/__adapt}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--adapter_approach /__}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--noise_mixing_strat /__noisem}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--softmax_temp /__temp}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--meta_loss_type /__mlstp}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--decouple_meta_and_model_updates /__decouple}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lora_r /__lora_r}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lora_alpha /__lora_alpha}"
ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--lora_dropout /__lora_dropout}"

if [[ "$ADD_ARGS_SAVE_NAME" == *"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"* ]]; then
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj/__target_all}"
elif [[ "$ADD_ARGS_SAVE_NAME" == *"q_proj v_proj"* ]]; then
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--target_modules q_proj v_proj/__target_qv}"
else
    target_tmp="${ADD_ARGS_SAVE_NAME#*--target_modules }"
    custom_targets="${target_tmp%% --*}"
    cleaned_targets="${custom_targets// /_}" 
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--target_modules $custom_targets/__target_modules_$cleaned_targets}"
fi

if [[ "$ADD_ARGS_SAVE_NAME" == *"embed_tokens score"* ]]; then
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--modules_to_save embed_tokens score/__save_emb_score}"
elif [[ "$ADD_ARGS_SAVE_NAME" == *"embed_tokens"* ]]; then
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--modules_to_save embed_tokens/__save_emb}"
else
    save_tmp="${ADD_ARGS_SAVE_NAME#*--modules_to_save }"
    custom_saves="${save_tmp%% --*}"
    cleaned_saves="${custom_saves// /_}" 
    ADD_ARGS_SAVE_NAME="${ADD_ARGS_SAVE_NAME//--modules_to_save $custom_saves/__save_$cleaned_saves}"
fi

TEMP_NAME=""
for word in $ADD_ARGS_SAVE_NAME; do
    if [[ $word == __* ]]; then
        TEMP_NAME+="$word"
    fi
done
ADD_ARGS_SAVE_NAME="$TEMP_NAME"

SAVE_FOLDER_NAME=${SELECTION_STRAT}${ADD_ARGS_SAVE_NAME}
echo "SAVE_FOLDER_NAME: $SAVE_FOLDER_NAME"

TRANSLATION_DIR=$DATA_DIR/translation
FASTALIGN_DIR=$DATA_DIR/fastalign
DICOALIGN_DIR=$DATA_DIR/dico-align
AWESOME_DIR=$DATA_DIR/awesome-align
RESULT_DIR=scripts/2026_emnlp/results/
SUB_DIR=$SAVE_FOLDER_NAME

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
uv run scripts/2026_emnlp/controlled_realignment.py \
    --translation_dir $TRANSLATION_DIR/$DATASET \
    --fastalign_dir $FASTALIGN_DIR/$DATASET \
    --dico_dir $DICOALIGN_DIR/$DATASET \
    --awesome_dir $AWESOME_DIR/$DATASET \
    --strategies $STRATEGY \
    --models $MODEL \
    --tasks $TASK \
    --cache_dir $CACHE_DIR \
    --n_epochs $n_epochs \
    --seed $SEED\
    --realignment_steps $MAX_REALIGNMENT_STEPS \
    --realignment_batch_size $REALIGNMENT_BATCH_SIZE\
    --right_langs $langs \
    --eval_langs $eval_langs \
    --output_file $RESULT_DIR/$SUB_DIR/${MODEL##*/}__${DATASET}__${STRATEGY}__${TASK}__${MAX_REALIGNMENT_STEPS}__${REALIGNMENT_BATCH_SIZE}.csv \
    --checkpoint_path $CKPT_DIR/$DATASET/$STRATEGY/$SUB_DIR/rebs_${REALIGNMENT_BATCH_SIZE} \
    --large_gpu $ADD_ARGS