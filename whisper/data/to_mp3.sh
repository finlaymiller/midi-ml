#!/bin/bash

# Directory containing the .aif files
DIRECTORY="."

# Loop through all .aif files in the directory
for file in "$DIRECTORY"/*.aif; do
    # Generate the new filename by replacing .aif with .wav
    newfile="${file%.aif}.mp3"

    # Convert from .aif to .mp3
    ffmpeg -i "$file" "$newfile"
done

echo "Conversion complete."

