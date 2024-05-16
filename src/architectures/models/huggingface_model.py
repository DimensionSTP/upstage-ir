from typing import Dict, Any, Union
import os

import torch
from torch import nn

from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        precision: Union[int, str],
        quantization_type: str,
        quantization_config: BitsAndBytesConfig,
        peft_type: str,
        peft_config: LoraConfig,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        if precision == 32 or precision == "32":
            self.precision = torch.float32
            self.attn_implementation = None
        elif precision == 16 or precision == "16":
            self.precision = torch.float16
            if "t5" in self.pretrained_model_name:
                self.attn_implementation = None
            else:
                self.attn_implementation = "flash_attention_2"
        elif precision == "bf16":
            self.precision = torch.bfloat16
            if "t5" in self.pretrained_model_name:
                self.attn_implementation = None
            else:
                self.attn_implementation = "flash_attention_2"
        else:
            self.precision = "auto"
            self.attn_implementation = None

        if quantization_type == "quantization":
            self.quantization_config = quantization_config
            self.quantization_config.bnb_4bit_compute_dtype = self.precision
            self.device_map = {
                "": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))
            }
            self.load_in_4bit = True
        elif quantization_type == "origin":
            self.quantization_config = None
            self.device_map = None
            self.load_in_4bit = False
        else:
            raise ValueError(f"Invalid quantization type: {quantization_type}.")

        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.precision,
                attn_implementation=self.attn_implementation,
                quantization_config=self.quantization_config,
                device_map=self.device_map,
                load_in_4bit=self.load_in_4bit,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.precision,
                attn_implementation=self.attn_implementation,
                quantization_config=self.quantization_config,
                device_map=self.device_map,
                load_in_4bit=self.load_in_4bit,
            )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )
        if quantization_type == "quantization":
            model = prepare_model_for_kbit_training(model)

        if peft_type == "lora":
            model.enable_input_require_grads()
            self.model = get_peft_model(model, peft_config)
        elif peft_type == "origin":
            self.model = model
        else:
            raise ValueError(f"Invalid PEFT type: {peft_type}.")

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        return output

    def generate(
        self,
        encoded: Dict[str, torch.Tensor],
        options: Dict[str, Any],
        target_max_length: int,
    ) -> torch.Tensor:
        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            options["max_length"] = target_max_length
        else:
            options["max_new_tokens"] = target_max_length
        output = self.model.generate(
            **{
                **encoded,
                **options,
            }
        )
        return output
