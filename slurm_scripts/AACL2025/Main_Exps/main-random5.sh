#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/aacl-main-random5-%j.out

#############################################################
# install the environment by loading in python and required packages

# module load StdEnv/2020 python/3.10.2 cuda/11.0 gcc/9.3.0 arrow/7.0.0
# # module load StdEnv/2023 python/3.12.4 cuda/12.2 arrow/17.0.0 - One possible combination for future env
# module load StdEnv/2023 python/3.10.13 cuda/12.2 arrow/17.0.0

source /home/bumie304/projects/def-annielee/bumie304/nlp_project/.venv/bin/activate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

export TRANSFORMERS_OFFLINE=1
for task in "xnli" "xtreme_r.udpos" "wikiann" "xquad"; do
    bash scripts/2025_aacl/run_langs_selection.sh "/home/bumie304/scratch/nlp_project/data" mix_opus100_nllb before_noaligner 31 random_langs_with_seed_5 $task
    bash scripts/2025_alignfreeze_continuation/distillation/run_langs_selection.sh data opus100 before_dico 31 random_lang_with_seed_14 udpos $task
done