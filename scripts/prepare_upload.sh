#!/bin/bash

path="src/preprocessing"
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="beomi"
model_type="OPEN-SOLAR-KO-10.7B"
quantization_type="origin"
peft_type="origin"
data_max_length=508
target_max_length=260
precision="bf16"
batch_size=32
epochs="30 32"

for epoch in $epochs
do
    python $path/prepare_upload.py \
        is_preprocessed=$is_preprocessed \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        quantization_type=$quantization_type \
        peft_type=$peft_type \
        data_max_length=$data_max_length \
        target_max_length=$target_max_length \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done
