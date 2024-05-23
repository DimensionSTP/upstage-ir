python src/preprocessing/merge_tokenizer.py upload_user=vicgalle model_type=SOLAR-13B-Instruct-v1.0
python src/preprocessing/preprocess_dataset.py mode=train upload_user=vicgalle model_type=SOLAR-13B-Instruct-v1.0
python src/preprocessing/preprocess_dataset.py mode=dev upload_user=vicgalle model_type=SOLAR-13B-Instruct-v1.0
python src/preprocessing/preprocess_dataset.py mode=test upload_user=vicgalle model_type=SOLAR-13B-Instruct-v1.0
