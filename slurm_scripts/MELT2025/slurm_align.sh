#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/align_freeze-%j.out

#############################################################
# install the environment by loading in python and required packages

module load StdEnv/2020 python/3.10.2 cuda/11.0 gcc/9.3.0 arrow/7.0.0
# module load StdEnv/2023 python/3.12.4 cuda/12.2 arrow/17.0.0 - One possible combination for future env
module load StdEnv/2023 python/3.10.13 cuda/12.2 arrow/17.0.0

source /home/bumie304/scratch/nlp_project/env/bin/activate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

bash scripts/2025_alignfreeze_continuation/distillation/run.sh data opus100 random_28 before_dico 66