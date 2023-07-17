#!/usr/bin/env bash

# Make Data Directory for Meme Text
DATA_DIR=${PWD}/Meme_Data
mkdir $DATA_DIR

# Copy all text files to Meme_Data folder
cp ${PWD}/data/*2.txt $DATA_DIR/

# Remove previous data folder
# rm -rf ${PWD}/data/

# # Convert .txt files to .json files
for file in $DATA_DIR/*.txt
    do 
        mv ${file} ${file/task2.txt/}.json
done

# Loop over all files in the directory
# ls | while read name
#     do
#         echo ${name}
# done