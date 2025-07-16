#!/bin/bash

set -e

OUTPUT_DIR=$1

langs=$2

for lang in $langs; do

    nllb_pair=$(python -c "print('-'.join(sorted(['eng_Latn', '$lang'])))")
    save_pair="${nllb_pair//eng_Latn/en}"
    echo "Pair: $nllb_pair → Save as: $save_pair"

    if [ ! -d "$OUTPUT_DIR/$save_pair" ]; then
        echo "Attempting to download $save_pair.gz from AllenNLP..."

        # Try primary download (AllenNLP)
        set +e
        wget https://storage.googleapis.com/allennlp-data-bucket/nllb/$nllb_pair.gz -O "$OUTPUT_DIR/$save_pair.gz"
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
    'zul_Latn': 'zu', 'ind_Latn': 'id', 'mya_Mymr': 'my', 'ayr_Latn': 'ay', 'grn_Latn': 'gn', 'quy_Latn': 'qu',
    'jav_Latn': 'jv', 'tgl_Latn': 'tl', 'tel_Telu': 'te', 'mar_Deva': 'mr', 'kaz_Cyrl': 'kz'
}
print(nllb_to_opus.get('$lang', '$lang'))
")

            # Compose OPUS-style pair
            opus_pair=$(python3 -c "print('-'.join(sorted(['en', '$opus_lang'])))")
            
            echo "Original NLLB pair: $nllb_pair"
            echo "Mapped OPUS pair: $opus_pair"

            # Try fallback download (OPUS)
            # if [ ! -f "$OUTPUT_DIR/$pair.txt.zip" ]; then
            #     wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$opus_pair.txt.zip -O "$OUTPUT_DIR/$pair.txt.zip"
            # fi
            wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/$opus_pair.txt.zip -O "$OUTPUT_DIR/$save_pair.txt.zip"

            unzip -o "$OUTPUT_DIR/$save_pair.txt.zip" -d "$OUTPUT_DIR/$save_pair"

            for file in "$OUTPUT_DIR/$save_pair"/NLLB.$opus_pair.*; do
                ext="${file##*.}"  # en, id, scores
                new_ext="$ext"

                # # Map file extension - dont need to map eng_Latn by now
                # if [ "$ext" == "en" ]; then
                #     new_ext="eng_Latn"
                if [ "$ext" == "$opus_lang" ]; then
                    new_ext="$lang"
                fi

                mv "$file" "$OUTPUT_DIR/$save_pair/NLLB.$save_pair.$new_ext"
            done
            rm "$OUTPUT_DIR/$save_pair.txt.zip"

        else
            echo "Primary download succeeded."

            mkdir -p "$OUTPUT_DIR/$save_pair/"
            gzip -d < "$OUTPUT_DIR/$save_pair.gz" > "$OUTPUT_DIR/$save_pair/$save_pair.tsv"

            lines=$(wc -l < "$OUTPUT_DIR/$save_pair/$save_pair.tsv")
            echo "$save_pair.tsv contains $lines sentences"

            awk -F'\t' '{print $1}' "$OUTPUT_DIR/$save_pair/$save_pair.tsv" > "$OUTPUT_DIR/$save_pair/NLLB.$save_pair.en"
            awk -F'\t' '{print $2}' "$OUTPUT_DIR/$save_pair/$save_pair.tsv" > "$OUTPUT_DIR/$save_pair/NLLB.$save_pair.$lang"

            rm "$OUTPUT_DIR/$save_pair.gz"
        fi
    fi
done