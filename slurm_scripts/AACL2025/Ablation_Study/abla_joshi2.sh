#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=outfile/aacl-abla-joshi2-%j.out

#############################################################
# install the environment by loading in python and required packages

source /home/bumie304/projects/def-annielee/bumie304/nlp_project/.venv/bin/activate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

export TRANSFORMERS_OFFLINE=1
for task in "xnli" "xtreme_r.udpos" "wikiann"; do
    bash scripts/2025_aacl/run_langs_selection.sh "/home/bumie304/scratch/nlp_project/data" mix_opus100_nllb before_noaligner 31 abla_most_uriel_joshi2 $task
    bash scripts/2025_aacl/run_langs_selection.sh "/home/bumie304/scratch/nlp_project/data" mix_opus100_nllb before_noaligner 31 abla_random_joshi2 $task
done
