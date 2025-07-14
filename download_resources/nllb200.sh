#!/bin/bash

set -e

OUTPUT_DIR=$1

langs=$2

for lang in $langs; do

    pair=$(python -c "print('-'.join(sorted(['eng_Latn', '$lang'])))")

    if [ ! -d "$OUTPUT_DIR/$pair" ]; then
        echo "Attempting to download $pair.gz from AllenNLP..."

        # Try primary download (AllenNLP)
        set +e
        wget https://storage.googleapis.com/allennlp-data-bucket/nllb/$pair.gz -O "$OUTPUT_DIR/$pair.gz"
        wget_status=$?
        set -e
        
        if [ $wget_status -ne 0 ]; then
            echo "Primary download failed. Falling back to OPUS..."

            # Mapping from NLLB lang to OPUS lang
            opus_lang=$(python3 -c "
nllb_to_opus = {
    'amh_Ethi': 'am', 'bam_Latn': 'bm', 'ewe_Latn': 'ewe', 'fon_Latn': 'fon', 'hau_Latn': 'ha',
    'ibo_Latn': 'ig', 'kin_Latn': 'rw', 'lin_Latn': 'ln', 'lug_Latn': 'lg', 'luo_Latn': 'luo',
    'mos_Latn': 'mos', 'nya_Latn': 'ny', 'gaz_Latn': 'om', 'sna_Latn': 'sn', 'swh_Latn': 'sw',
    'tsn_Latn': 'tn', 'twi_Latn': 'tw', 'wol_Latn': 'wo', 'xho_Latn': 'xh', 'yor_Latn': 'yo',
    'zul_Latn': 'zu', 'ind_Latn': 'id', 'mya_Mymr': 'my', 'ayr_Latn': 'ay', 'grn_Latn': 'gn', 'quy_Latn': 'qu'
}
print(nllb_to_opus.get('$lang', '$lang'))
")

            # Compose OPUS-style pair
            opus_pair=$(python3 -c "print('-'.join(sorted(['en', '$opus_lang'])))")
            
            echo "Original NLLB pair: $pair"
            echo "Mapped OPUS pair: $opus_pair"

            # Try fallback download (OPUS)
            # if [ ! -f "$OUTPUT_DIR/$pair.txt.zip" ]; then
            #     wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$opus_pair.txt.zip -O "$OUTPUT_DIR/$pair.txt.zip"
            # fi
            wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$opus_pair.txt.zip -O "$OUTPUT_DIR/$pair.txt.zip"

            unzip -o "$OUTPUT_DIR/$pair.txt.zip" -d "$OUTPUT_DIR/$pair"

            for file in "$OUTPUT_DIR/$pair"/NLLB.$opus_pair.*; do
                ext="${file##*.}"  # en, id, scores
                new_ext="$ext"

                # Map file extension
                if [ "$ext" == "en" ]; then
                    new_ext="eng_Latn"
                elif [ "$ext" == "$opus_lang" ]; then
                    new_ext="$lang"
                fi

                mv "$file" "$OUTPUT_DIR/$pair/NLLB.$pair.$new_ext"
            done
            rm "$OUTPUT_DIR/$pair.txt.zip"

        else
            echo "Primary download succeeded."

            mkdir -p "$OUTPUT_DIR/$pair/"
            gzip -d < "$OUTPUT_DIR/$pair.gz" > "$OUTPUT_DIR/$pair/$pair.tsv"

            lines=$(wc -l < "$OUTPUT_DIR/$pair/$pair.tsv")
            echo "$pair.tsv contains $lines sentences"

            awk -F'\t' '{print $1}' "$OUTPUT_DIR/$pair/$pair.tsv" > "$OUTPUT_DIR/$pair/NLLB.$pair.eng_Latn"
            awk -F'\t' '{print $2}' "$OUTPUT_DIR/$pair/$pair.tsv" > "$OUTPUT_DIR/$pair/NLLB.$pair.$lang"

            rm "$OUTPUT_DIR/$pair.gz"
        fi
    fi
done