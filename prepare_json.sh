#!/bin/bash

# Define the source and target substrings
source_substring1="/local_disk/ether/ylabrak/SpokenMedicalQA/audios_wav_ffmpeg_train_concat"
target_substring1="$HOME/SpokenMedicalQA_data/audios_wav_ffmpeg_train_concat"

source_substring2="/local_disk/ether/ylabrak/SpokenMedicalQA/audios_wav_ffmpeg_concat"
target_substring2="$HOME/SpokenMedicalQA_data/audios_wav_ffmpeg_concat"

# Loop through all CSV files in the specified directory
for file in $HOME/SpokenMedicalQA_data/data-30+json/*.json; do
    # Create a temporary file to store modified content
    temp_file=$(mktemp)

    # Replace substrings in each line and write to the temporary file
    awk -v source1="$source_substring1" -v target1="$target_substring1" \
        -v source2="$source_substring2" -v target2="$target_substring2" \
        'BEGIN {FS=OFS=","} {gsub(source1, target1); gsub(source2, target2); print}' "$file" > "$temp_file"

    # Replace the original file with the modified content
    mv "$temp_file" "$file"

    echo "File '$file' updated."
done

echo "Script completed."

