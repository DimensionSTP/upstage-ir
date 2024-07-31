#!/bin/bash

path="src/preprocessing"

python $path/add_empty_column.py
python $path/make_corpus.py
python $path/train_tokenizer.py
python $path/merge_tokenizer.py
python $path/merge_model.py

dataset_modes="train"

for dataset_mode in $dataset_modes
do
    python $path/preprocess_dataset.py mode=$dataset_mode
done
