#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=20G
#SBATCH --time=120:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quangphuoc.nguyen@ontariotechu.net
#SBATCH --output=extract_embeddings_and_calculate_ani-%j.out

#############################################################
# install the environment by loading in python and required packages
module load StdEnv/2020 python/3.10.2 cuda/11.0 gcc/9.3.0 arrow/7.0.0
source /home/bumie304/scratch/nlp_project/env/bin/activate

#############################################################
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
python scripts/2025_alignfreeze_continuation/distillation/anisotropy/extract_embeddings.py \
    --strategies freeze_ffn_dico freeze_ffn_realign_random_onethird_dico random_onethird_dico baseline before_dico \
    --strategy_dir /home/bumie304/scratch/results/ \
    --save-dir /home/bumie304/scratch/nlp_project/embs/ \

python scripts/2025_alignfreeze_continuation/distillation/anisotropy/calculate_anisotropy.py \
    --embs-dir /home/bumie304/scratch/nlp_project/embs \
    --strategies freeze_ffn_dico freeze_ffn_realign_random_onethird_dico random_onethird_dico baseline before_dico