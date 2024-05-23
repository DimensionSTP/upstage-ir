python main.py mode=predict is_tuned=untuned is_preprocessed=True quantization_type=quantization peft_type=lora epoch={epoch}
python merge_predictions.py is_tuned=untuned is_preprocessed=True quantization_type=quantization peft_type=lora epoch={epoch}
