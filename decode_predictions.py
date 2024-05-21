import os
import pickle
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
)
def decode_predictions(
    config: DictConfig,
) -> None:
    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = "Summarize the dialogues of two people appropriately."
        prompt = f"""### Instruction:
        {default_system_prompt} 

        ### Input:
        {data.strip()}

        ### Response:
        """.strip()
        return prompt

    with open(
        f"{config.connected_dir}/preds/{config.pred_name}.pickle",
        "rb",
    ) as f:
        all_predictions = pickle.load(f)
    generation_df = pd.read_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv"
    )
    generation_df["prompt"] = generation_df["dialogue"].apply(generate_prompt)
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_df["encoded_prompt_length"] = generation_df["prompt"].apply(
        lambda x: len(
            tokenizer(
                x,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        )
    )
    decoded_predictions = []
    for i in range(len(all_predictions[0])):
        encoded_prompt_length = generation_df.loc[i, "encoded_prompt_length"]
        summary_prediction = all_predictions[i, encoded_prompt_length:]
        decoded_prediction = tokenizer.batch_decode(
            sequences=summary_prediction,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_predictions.append(decoded_prediction)
    generation_df = pd.read_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv"
    )
    generation_df[config.target_column_name] = decoded_predictions
    generation_df.to_csv(
        f"{config.connected_dir}/submissions/from_decoded_predictions/{config.pred_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    decode_predictions()
