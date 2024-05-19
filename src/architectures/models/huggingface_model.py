from typing import Dict, Any, Union
import os

import torch
from torch import nn

from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        precision: Union[int, str],
        mode: str,
        quantization_type: str,
        quantization_config: BitsAndBytesConfig,
        peft_type: str,
        peft_config: LoraConfig,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name

        self.attn_implementation = None
        if precision == 32 or precision == "32":
            self.precision = torch.float32
        elif precision == 16 or precision == "16":
            self.precision = torch.float16
            if "t5" not in self.pretrained_model_name:
                self.attn_implementation = "flash_attention_2"
        elif precision == "bf16":
            self.precision = torch.bfloat16
            if "t5" not in self.pretrained_model_name:
                self.attn_implementation = "flash_attention_2"
        else:
            self.precision = "auto"

        self.mode = mode
        self.quantization_type = quantization_type
        self.quantization_config = None
        self.device_map = None
        if self.quantization_type == "quantization":
            self.quantization_config = quantization_config
            if self.mode in ["test" "predict"]:
                self.quantization_config.load_in_4bit = False
            else:
                self.quantization_config.load_in_4bit = True
            self.quantization_config.bnb_4bit_compute_dtype = self.precision
            self.device_map = {
                "": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))
            }
        if self.quantization_type not in ["origin", "quantization"]:
            raise ValueError(f"Invalid quantization type: {self.quantization_type}.")

        self.peft_type = peft_type
        self.peft_config = peft_config
        if self.mode in ["test" "predict"]:
            self.peft_config.inference_mode = True

        self.model = self.get_model()

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
        target_min_length: int,
    ) -> torch.Tensor:
        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            options["max_length"] = target_max_length
        else:
            options["max_new_tokens"] = target_max_length
            options["min_new_tokens"] = target_max_length
        output = self.model.generate(
            **{
                **encoded,
                **options,
            }
        )
        return output

    def get_model(self) -> PreTrainedModel:
        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.precision,
                attn_implementation=self.attn_implementation,
                quantization_config=self.quantization_config,
                device_map=self.device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.precision,
                attn_implementation=self.attn_implementation,
                quantization_config=self.quantization_config,
                device_map=self.device_map,
            )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

        if self.quantization_type == "quantization" and self.mode not in [
            "test",
            "predict",
        ]:
            model = prepare_model_for_kbit_training(model)

        if self.peft_type == "lora":
            model.enable_input_require_grads()
            model = get_peft_model(model, self.peft_config)
        if self.peft_type not in ["origin", "lora"]:
            raise ValueError(f"Invalid PEFT type: {self.peft_type}.")
        return model
