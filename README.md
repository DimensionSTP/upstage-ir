# Summary DL pipeline for NLP competition

## For Upstage NLP competition

### Dataset
Upstage NLP competition dataset(dialogue summarization)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/upstage-nlp.git
cd upstage-nlp

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Model Hyper-Parameters Tuning

* end-to-end
```shell
python main.py mode=tune is_tuned=untuned num_trials={num_trials}
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={tuned or untuned} num_trials={num_trials}
```

### Test

* end-to-end
```shell
python main.py mode=test is_tuned={tuned or untuned} num_trials={num_trials} epoch={ckpt epoch}
```

### Prediction

* end-to-end
```shell
python main.py mode=predict is_tuned={tuned or untuned} num_trials={num_trials} epoch={ckpt epoch}
```

### Additional Options

* pure decoder based LLM QLoRA 4-bit quantization option
```shell
quantization_type={origin or quantization} 
```

* pure decoder based LLM LoRA or QLoRA PEFT option
```shell
peft_type={origin or lora}
```

* for LLM fine-tuning in multi-GPU, recommended
```shell
strategy={deepspeed_stage_2 or deepspeed_stage_2_offload}
```

* upload user name and model name at HuggingFace Model card
```shell
upload_user={upload_user} 
model_type={model_type}
```

* Set data and target max lengt for model training and generation
```shell
data_max_length={data_max_length} 
target_max_length={target_max_length} 
```


__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__