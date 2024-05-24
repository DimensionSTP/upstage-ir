path="src/preprocessing"

upload_user="vicgalle"
model_type="SOLAR-13B-Instruct-v1.0"

python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type

dataset_modes="train dev test"

for dataset_mode in $dataset_modes
do
    python $path/preprocess_dataset.py mode=$dataset_mode upload_user=$upload_user model_type=$model_type
done
