#!/bin/bash

set -e

DATA_DIR=$1

# https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py
allenai_langs="amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
# lacking Ghomala (bbj), Nigerian Pidgin (pcm) | Ewe (ee) 4.4M sentences but cannot be downloaded through OPUS 

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
NLLB_DIR=$DATA_DIR/nllb200
TRANSLATION_DIR=$DATA_DIR/translation

mkdir -p $CACHE_DIR
mkdir -p $NLLB_DIR
mkdir -p $TRANSLATION_DIR

export DATA_DIR=$DATA_DIR
export NLLB_DIR=$NLLB_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR

# download NLLB 200
echo "download NLLB 200"
bash download_resources/nllb200.sh $NLLB_DIR "$allenai_langs"

for lang in $allenai_langs; do
    echo "parsing lang $lang for nllb-200"

    mkdir -p $TRANSLATION_DIR/nllb200

    # For allenai url langs
    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    uv run subscripts/prepare_pharaoh_dataset.py \
        $NLLB_DIR/$pair/NLLB.$pair.en \
        $NLLB_DIR/$pair/NLLB.$pair.$lang \
        $TRANSLATION_DIR/nllb200/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang

done
