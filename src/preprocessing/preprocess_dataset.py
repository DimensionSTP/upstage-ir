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
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}"
    )

    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = "너의 역할은 대화 내용을 요약해주는 요약 전문가야. 다음 사람들의 대화 내용을 보고 적절히 요약해줘."
        prompt = f"""### Instruction:
        {default_system_prompt} 
        ### Input:
        {data.strip()}
        ### Response:
        """.strip()
        return prompt

    df["prompt"] = df["dialogue"].apply(generate_prompt)

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

    df["cut_prompt"] = df["prompt"].apply(
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
