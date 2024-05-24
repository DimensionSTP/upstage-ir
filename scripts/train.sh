quantization_type="quantization"
peft_type="lora"
upload_user="vicgalle"
model_type="SOLAR-13B-Instruct-v1.0"

python main.py mode=train is_tuned=untuned is_preprocessed=True quantization_type=$quantization_type peft_type=$peft_type upload_user=$upload_user model_type=$model_type
