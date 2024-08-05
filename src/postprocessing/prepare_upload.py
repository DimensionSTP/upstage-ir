import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import re
import json

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from safetensors.torch import save_file

from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def prepare_upload(
    config: DictConfig,
) -> None:
    save_dir = f"{config.connected_dir}/prepare_upload/{config.pretrained_model_name}/epoch={config.epoch}"
    if config.strategy.startswith("deepspeed"):
        checkpoint = torch.load(f"{config.ckpt_path}/model.pt")
    else:
        checkpoint = torch.load(config.ckpt_path)
    checkpoint_state_dict = checkpoint["state_dict"]
    model_state_dict = {}
    for k, v in list(checkpoint_state_dict.items()):
        if k.startswith("model."):
            k = re.sub(
                r"(model\.)+(.*)",
                r"model.\2",
                k,
            )
            if k.startswith("model.lm_head"):
                k = k.replace(
                    "model.",
                    "",
                )
            model_state_dict[k] = v

    if config.is_preprocessed:
        if os.path.exists(config.merged_model_path):
            model_path = config.merged_model_path
        else:
            model_path = config.pretrained_model_name

        if os.path.exists(config.custom_data_encoder_path):
            data_encoder_path = config.custom_data_encoder_path
        else:
            data_encoder_path = config.pretrained_model_name

    original_model = AutoModelForCausalLM.from_pretrained(model_path)
    original_model.load_state_dict(model_state_dict)
    state_dict = original_model.state_dict()
    if config.precision == 32 or config.precision == "32":
        safetensors_dtype = torch.float32
        torch_dtype = "float32"
    elif config.precision == 16 or config.precision == "16":
        safetensors_dtype = torch.float16
        torch_dtype = "float16"
    elif config.precision == "bf16":
        safetensors_dtype = torch.bfloat16
        torch_dtype = "bfloat16"
    else:
        raise ValueError(f"Invalid precision type: {config.precision}")
    keys = list(state_dict.keys())
    num_splits = config.num_safetensors
    split_size = len(keys) // num_splits
    total_size = sum(
        param.numel() * param.element_size() for param in state_dict.values()
    )
    index_dict = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": {},
    }

    if not os.path.exists(save_dir):
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
    for i in tqdm(range(num_splits)):
        safe_tensors_name = f"model-{i+1:05d}-of-{num_splits:05d}.safetensors"
        part_state_dict = {
            k: state_dict[k].to(safetensors_dtype)
            for k in keys[i * split_size : (i + 1) * split_size]
        }
        part_state_dict_mapping = {
            k: safe_tensors_name for k in keys[i * split_size : (i + 1) * split_size]
        }
        index_dict["weight_map"].update(part_state_dict_mapping)
        save_file(
            part_state_dict,
            f"{save_dir}/{safe_tensors_name}",
            metadata={
                "format": "pt",
            },
        )
    with open(f"{save_dir}/model.safetensors.index.json", "w") as f:
        json.dump(
            index_dict,
            f,
            indent=2,
        )
    tokenizer = AutoTokenizer.from_pretrained(data_encoder_path)
    tokenizer.save_pretrained(save_dir)
    model_config = AutoConfig.from_pretrained(model_path)
    model_config._name_or_path = (
        f"{config.user_name}/{config.model_type}-{config.upload_tag}"
    )
    model_config.torch_dtype = torch_dtype
    if config.is_preprocessed:
        model_config.vocab_size = tokenizer.vocab_size
    model_config.save_pretrained(save_dir)


if __name__ == "__main__":
    prepare_upload()
