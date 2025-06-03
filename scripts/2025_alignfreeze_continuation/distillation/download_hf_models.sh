#!/bin/bash

CACHE_DIR="/home/bumie304/scratch/nlp_project/cache/transformers"

mkdir -p "$CACHE_DIR"

MODELS=(
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
    "distilbert-base-multilingual-cased"
)

for MODEL in "${MODELS[@]}"; do
    echo "Downloading $MODEL..."
    huggingface-cli download "$MODEL" --cache-dir "$CACHE_DIR" &
    # huggingface-cli download "$MODEL" \
    # --local-dir $CACHE_DIR/models--$MODEL \
    # --local-dir-use-symlinks False
done

# Wait for all background jobs to finish
wait

echo "✅ All models downloaded to $CACHE_DIR"
