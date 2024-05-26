#!/bin/bash

path="src/preprocessing"
upload_users=("vicgalle" "beomi" "beomi" "beomi" "Bllossom" "upstage")
model_types=("SOLAR-13B-Instruct-v1.0" "OPEN-SOLAR-KO-10.7B" "Llama-3-Open-Ko-8B" "Llama-3-Open-Ko-8B-Instruct-preview" "llama-3-Korean-Bllossom-70B" "SOLAR-0-70b-8bit")
length=${#model_types[@]}
dataset_modes="train dev test"

for ((i=0; i<$length; i++))
do
    upload_user=${upload_users[$i]}
    model_type=${model_types[$i]}
    python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
    for dataset_mode in $dataset_modes
    do
        python $path/preprocess_dataset.py mode=$dataset_mode upload_user=$upload_user model_type=$model_type
    done
done
