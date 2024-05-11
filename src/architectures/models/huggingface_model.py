from typing import Dict

import torch
from torch import nn

from transformers import BartForConditionalGeneration


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
    ) -> None:
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            output_hidden_states=False,
        )

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        return output

    def generate(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        output = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        return output
