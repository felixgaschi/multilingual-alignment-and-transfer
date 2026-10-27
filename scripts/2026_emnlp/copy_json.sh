#!/bin/bash

JSON_SOURCE=$1
DEST_DIR=$2

# --- PHASE 1: Dry Run ---
echo "--- DRY RUN: The following files would be moved ---"
found_files=0

for dir in "$JSON_SOURCE"/*/; do
    subdir=$(basename "$dir")
    if [ -d "$DEST_DIR/$subdir" ]; then
        # Check if there are actually .jsonl files to move
        ls "$dir"*.jsonl >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "From: $dir*.jsonl"
            echo "To:   $DEST_DIR/$subdir/"
            found_files=$((found_files + 1))
        fi
    fi
done

if [ $found_files -eq 0 ]; then
    echo "No matching files or directories found. Exiting."
    exit 0
fi

# --- PHASE 2: User Confirmation ---
echo ""
read -p "Do you want to proceed with the copy? (y/n): " confirm

if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    echo "Starting copy..."
    for dir in "$JSON_SOURCE"/*/; do
        subdir=$(basename "$dir")
        if [ -d "$DEST_DIR/$subdir" ]; then
            cp "$dir"*.jsonl "$DEST_DIR/$subdir/" 2>/dev/null
        fi
    done
    echo "Done!"
else
    echo "Operation cancelled."
fi