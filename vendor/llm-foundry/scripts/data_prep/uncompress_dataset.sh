#!/bin/bash

SOURCE_DIR="/mnt/fl/datasets/pretrain/compressed_sources/3.5"
DEST_DIR="/mnt/fl/uncompressed_sources/sources/3.5"

NCORES=8

export SOURCE_DIR
export DEST_DIR

# Create destination directory if it does not exist.
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi

# Define uncompression function. It strips first components of the file path
# (we have absolute paths in archive) to extract files to the same relative
# directory structure in destination folder.
uncompress_file() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
          exit
  fi

  # Strip first 6 components. Used for the numbered set processing (3.5, test_data).
  tar -xf "$file" -C "$DEST_DIR" --strip-components=6

  # Strip first 7 components. Used for processing specific dataset (3.5/wikipedia, test_data/).
  # tar -xf "$file" -C "$DEST_DIR" --strip-components=7
}

export -f uncompress_file

# Determine source and target files and get unprocessed files. Strip .tar.gz suffix from the
# source file and then append it back when we pass it to uncompress function.
source_files=$(find $SOURCE_DIR -type f | sort | sed "s|^$SOURCE_DIR/||")
target_files=$(find $DEST_DIR -type f | sort | sed "s|^$DEST_DIR/||")

unprocessed_files=$(comm -23 <(echo "$source_files" | sed "s/.tar.gz$//") <(echo "$target_files"))

echo "Unprocessed files: $unprocessed_files"
echo

echo "$unprocessed_files" | parallel -j $NCORES --bar --no-notice uncompress_file "$SOURCE_DIR/{}.tar.gz"
