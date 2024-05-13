from typing import Dict, Any

import torch
from torch import nn

from transformers import (
    BitsAndBytesConfig,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        quantization_type: str,
        quantization_config: BitsAndBytesConfig,
        peft_type: str,
        peft_config: LoraConfig,
    ) -> None:
        super().__init__()
        if quantization_type == "quantization":
            self.quantization_config = quantization_config
        elif quantization_type == "origin":
            self.quantization_config = None
        else:
            raise ValueError(f"Invalid quantization type: {quantization_type}.")

        if "bart" in pretrained_model_name:
            model = BartForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
                quantization_config=self.quantization_config,
            )
        elif "t5" in pretrained_model_name:
            model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
                quantization_config=self.quantization_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                output_hidden_states=False,
                quantization_config=self.quantization_config,
            )

        if quantization_type == "quantization":
            model = prepare_model_for_kbit_training(model)

        if peft_type == "lora":
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
    ) -> torch.Tensor:
        output = self.model.generate(
            **{
                **encoded,
                **options,
            }
        )
        return output
