quantization_type="quantization"
peft_type="lora"
upload_user="vicgalle"
model_type="SOLAR-13B-Instruct-v1.0"
epoch=1

python main.py mode=predict is_tuned=untuned is_preprocessed=True quantization_type=$quantization_type peft_type=$peft_type upload_user=$upload_user model_type=$model_type epoch=$epoch
python merge_predictions.py is_tuned=untuned is_preprocessed=True quantization_type=$quantization_type peft_type=$peft_type upload_user=$upload_user model_type=$model_type epoch=$epoch
python decode_predictions.py is_tuned=untuned is_preprocessed=True quantization_type=$quantization_type peft_type=$peft_type upload_user=$upload_user model_type=$model_type epoch=$epoch
