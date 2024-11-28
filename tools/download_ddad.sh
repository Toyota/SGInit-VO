#!/bin/bash
# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

ROOT_DIR="/data/datasets"
FOLDER_NAME="DDAD"
TAR_URL="https://tri-ml-public.s3.amazonaws.com/github/vidar/datasets/full/DDAD.tar"
TAR_FILE="$ROOT_DIR/$(basename $TAR_URL)"

# Check if the folder exists
if [ -d "$ROOT_DIR/$FOLDER_NAME" ]; then
  echo "Folder $FOLDER_NAME already exists in $ROOT_DIR. No action taken."
else
  echo "Folder $FOLDER_NAME does not exist. Downloading and extracting..."
  
  # Download the tar file if it doesn't already exist
  if [ ! -f "$TAR_FILE" ]; then
    wget "$TAR_URL" -O "$TAR_FILE"
  else
    echo "Tar file already exists at $TAR_FILE. Skipping download."
  fi
  
  # Extract the downloaded tar file
  tar -xf "$TAR_FILE" -C "$ROOT_DIR"

  # Optional: Clean up by removing the tar file after extraction
  # rm "$TAR_FILE"
fi