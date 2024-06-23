import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_dataset(
    config: DictConfig,
) -> None:
    df = pd.read_csv(f"{config.connected_dir}/data/{config.mode}.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}",
        use_fast=True,
    )

    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = """
너는 과학 질문에 대한 답변을 제공하는 챗봇이야.
너의 역할은 사용자들이 과학적 주제에 대해 궁금해하는 질문에 명확하고 정확한 답변을 제공하는 거야.
"""
        prompt = f"""### Instruction:
{default_system_prompt} 

### Input(질문):
{data.strip()}

### Response(답변):
""".strip()
        return prompt

    df["prompt"] = df[config.data_column_name].apply(generate_prompt)
    df[config.data_column_name] = df[config.data_column_name].apply(lambda x: x.strip())

    def cut_prompt_to_length(
        prompt: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> str:
        tokens = tokenizer.tokenize(prompt)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        cut_prompt = tokenizer.convert_tokens_to_string(tokens)
        return cut_prompt

    df[config.prompt_column_name] = df["prompt"].apply(
        lambda x: cut_prompt_to_length(
            prompt=x,
            tokenizer=tokenizer,
            max_length=config.data_max_length,
        )
    )
    if not os.path.exists(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}"
    ):
        os.makedirs(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}",
            exist_ok=True,
        )
    df.to_csv(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}/{config.mode}.csv",
        index=False,
    )


if __name__ == "__main__":
    preprocess_dataset()
