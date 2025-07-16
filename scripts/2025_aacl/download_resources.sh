#!/bin/bash

set -e

DATA_DIR=$1

opus100_langs="af ar az bg bn de el es et eu fa fi fr gu he hi hu id it ja ka ko lt ml ms nl pa pl pt ro ru ta th tr uk ur vi zh"
# https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py
nllb_langs="amh_Ethi bam_Latn ewe_Latn fon_Latn hau_Latn ibo_Latn kin_Latn lin_Latn lug_Latn luo_Latn mos_Latn nya_Latn gaz_Latn sna_Latn swh_Latn tsn_Latn twi_Latn wol_Latn xho_Latn yor_Latn zul_Latn mya_Mymr jav_Latn tgl_Latn tel_Telu mar_Deva kaz_Cyrl"
# lacking Ghomala (bbj), Nigerian Pidgin (pcm) | Ewe (ee) 4.4M sentences but cannot be downloaded through OPUS 

mkdir -p $DATA_DIR

CACHE_DIR=$DATA_DIR/cache/datasets
OPUS_DIR=$DATA_DIR/opus100
NLLB_DIR=$DATA_DIR/nllb200
TRANSLATION_DIR=$DATA_DIR/translation

mkdir -p $CACHE_DIR
mkdir -p $OPUS_DIR
mkdir -p $NLLB_DIR
mkdir -p $TRANSLATION_DIR

export DATA_DIR=$DATA_DIR
export OPUS_DIR=$OPUS_DIR
export NLLB_DIR=$NLLB_DIR
export TRANSLATION_DIR=$TRANSLATION_DIR

# download OPUS 100
echo "download OPUS 100"
bash download_resources/opus100.sh $OPUS_DIR "$opus100_langs"

# download NLLB 200
echo "download NLLB 200"
bash download_resources/nllb200.sh $NLLB_DIR "$nllb_langs"

# Tokenize sentences
mkdir -p $TRANSLATION_DIR/mix_opus100_nllb
for lang in $opus100_langs; do
    echo "parsing lang $lang for opus-100"

    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    uv run subscripts/prepare_pharaoh_dataset.py \
        $OPUS_DIR/$pair/opus.$pair-train.en \
        $OPUS_DIR/$pair/opus.$pair-train.$lang \
        $TRANSLATION_DIR/mix_opus100_nllb/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang
done

for lang in $nllb_langs; do
    echo "parsing lang $lang for nllb-200"

    # For allenai url langs
    pair=$(python -c "print('-'.join(sorted(['en', '$lang'])))")

    # Create FastAlign-compatible tokenized translation dataset
    uv run subscripts/prepare_pharaoh_dataset.py \
        $NLLB_DIR/$pair/NLLB.$pair.en \
        $NLLB_DIR/$pair/NLLB.$pair.$lang \
        $TRANSLATION_DIR/mix_opus100_nllb/en-$lang.tokenized.train.txt \
        --left_lang en --right_lang $lang

done
