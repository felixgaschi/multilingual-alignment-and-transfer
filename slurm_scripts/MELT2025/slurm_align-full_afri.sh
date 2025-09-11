#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/align_freeze-full_afri-%j.out

#############################################################
# install the environment by loading in python and required packages

module load StdEnv/2020 python/3.10.2 cuda/11.0 gcc/9.3.0 arrow/7.0.0
source /home/bumie304/scratch/nlp_project/env/bin/activate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

# bash scripts/2025_alignfreeze_continuation/alignfreeze/download_resources_nllb200.sh "/home/bumie304/scratch/nlp_project/data"

#31,42,66,23,17
bash scripts/2025_alignfreeze_continuation/distillation/run_langs_selection.sh "/home/bumie304/scratch/nlp_project/data" nllb200 before_noaligner 31 full_afri xnli

bash scripts/2025_alignfreeze_continuation/distillation/run_langs_selection.sh "/home/bumie304/scratch/nlp_project/data" nllb200 before_noaligner 31 full_afri udpos