#!/bin/bash

# Directory containing the .aif files
DIRECTORY="data/talks"

# Loop through all .aif files in the directory
for file in "$DIRECTORY"/*.aif; do
    # Generate the new filename by replacing .aif with .wav
    newfile="${file%.aif}.wav"

    # Convert from .aif to .wav
    ffmpeg -i "$file" "$newfile"
done

echo "Conversion complete."

