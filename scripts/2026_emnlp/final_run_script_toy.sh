#!/bin/bash

# Toy version of final_run_script.sh, matched to download_resources_toy.sh.
# Uses the same 10-language subset for realignment (6 opus100 + 4 NLLB) and
# evaluates each task only on the languages we actually have data for.

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
# Toy realignment set: the 10 langs downloaded by download_resources_toy.sh
TOY_LANGS="de fr hi vi id eu swh_Latn hau_Latn yor_Latn mya_Mymr"

if [ "$SELECTION_STRAT" == "xt_afri" ]; then
    langs="$TOY_LANGS"
elif [ "$SELECTION_STRAT" == "most_uriel_en_40" ]; then
    langs="$TOY_LANGS"
elif [ "$STRATEGY" == "baseline" ]; then
    langs="$TOY_LANGS"
    echo "Fine-tuning baseline."
    SELECTION_STRAT="ft_only"
else
    echo "Unknown SELECTION_STRAT value: $SELECTION_STRAT. Exitting."
    exit 1
fi
#********************************************END REALIGNMENT LANGUAGE SETTING********************************************

#********************************************TASK SETTING********************************************
# Eval langs restricted to those we downloaded (with code mappings):
#   id <-> ind, fr <-> fra, swh_Latn -> swa, mya_Mymr -> mya/my, yor_Latn -> yor/yo
if [ "$TASK" == "xnli" ]; then
    n_epochs=2
    eval_langs="de fr fra hi vi ind hau swa yor mya"
elif [ "$TASK" == "wikiann" ]; then
    n_epochs=5
    eval_langs="de eu fr hi id vi my hau swa yo yor"
elif [ "$TASK" == "xtreme_r.udpos" ]; then
    n_epochs=5
    eval_langs="de eu fr hi id vi hau swa yo yor"
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

ADD_ARGS_SAVE_NAME="${ADD_ARGS//--enable_weighted_sampling/__weighted}"
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
SUB_DIR=toy/$SAVE_FOLDER_NAME

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
