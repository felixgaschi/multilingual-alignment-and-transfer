#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/aacl-all_main-%j.out

#############################################################
# install the environment by loading in python and required packages

# NOTE: Must change into project dir to run the scipt
cur_dir=$(pwd)
source "$cur_dir/.venv/bin/activate"

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

DATA_DIR=~/scratch/nlp_project/data
DATASET=mix_opus100_nllb
STRATEGY=before_noaligner
SEED=31

export TRANSFORMERS_OFFLINE=1
for task in "xnli" "xtreme_r.udpos" "wikiann" "xquad"; do
    #===============================BASELINE===============================
    for baseline in "xt_afri" "xt_only" "afri_only"; do
        bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED $baseline $task
    done
    #===============================URIEL_DIVERSITY===============================
    for uriel_exp in "most_uriel_en" "least_uriel_en"; do
        for size in 5 10 20 40; do
            bash scripts/2025_aacl/run_langs_selection.sh $DATA_DIR $DATASET $STRATEGY $SEED "${uriel_exp}_${size}" $task
        done
    done
    #===============================FAMILY_DIVERSITY===============================
    for family_exp in "most_family_en" "least_family_en"; do
        for size in 5 10 20 25; do
            bash scripts/2025_aacl/run_langs_selection.sh $DATA_DIR $DATASET $STRATEGY $SEED "${family_exp}_${size}" $task
        done
    done
    #===============================SCRIPT_DIVERSITY===============================
    for most_distinct_script_size in 5 10 18; do
        bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED "$most_distinct_script_${most_distinct_script_size}" $task
    done
    for most_same_script_size in 5 10 20 41; do
        bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED "$most_same_script_${most_same_script_size}" $task
    done
    for least_same_script_size in 5 10 20; do
        bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED "$least_same_script_${least_same_script_size}" $task
    done
    #===============================RANDOM===============================
    for random_size in 5 10 20 40; do
        bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED "$random_langs_with_seed_${random_size}" $task
    done
    #===============================ABLATION_10===============================
    for lang_class in "joshi45" "joshi3" "joshi35" "joshi2" "seen_xlmr" "unseen_xlmr" "seen_mbert" "unseen_mbert"; do
        # Dont have most script and most family yet
        for strat in "random" "most_uriel"; do
            bash scripts/2025_aacl/run_langs_selection.sh  $DATA_DIR  $DATASET  $STRATEGY  $SEED "$abla_${strat}_${lang_class}" $task
        done
    done
done

