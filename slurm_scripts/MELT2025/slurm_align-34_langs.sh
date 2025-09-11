#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/align_freeze-34_langs-%j.out

#############################################################
# install the environment by loading in python and required packages

module load StdEnv/2020 python/3.10.2 cuda/11.0 gcc/9.3.0 arrow/7.0.0
source /home/bumie304/scratch/nlp_project/env/bin/activate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

#31,42,66,23,17
# bash scripts/2025_alignfreeze_continuation/distillation/run_langs_selection.sh data opus100 before_dico 31 34_langs xnli

bash scripts/2025_alignfreeze_continuation/distillation/run_langs_selection.sh data opus100 before_dico 31 34_langs udpos