#!/bin/bash

# Toy version of download_resources.sh: downloads a small, diverse subset of
# languages for quick smoke-testing of the NLI, NER (wikiann), and UDPOS
# evaluations in final_run_script.sh.
#
# Language picks (avoiding zh / th):
#   opus100: de fr (high-resource), hi vi id (mid-resource), eu (low, Basque)
#   nllb200: swh_Latn hau_Latn yor_Latn (low African), mya_Mymr (low, Burmese)
#
# Each evaluation task ends up with languages spanning multiple resource tiers,
# weighted toward low-resource where possible.

set -e

DATA_DIR=$1

opus100_langs="de fr hi vi id eu"
# https://huggingface.co/datasets/allenai/nllb/blob/main/nllb_lang_pairs.py
nllb_langs="swh_Latn hau_Latn yor_Latn mya_Mymr"

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

rm -rf $OPUS_DIR

# download NLLB 200
echo "download NLLB 200"
bash download_resources/nllb200.sh $NLLB_DIR "$nllb_langs"

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

rm -rf $NLLB_DIR