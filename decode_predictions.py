import dotenv

dotenv.load_dotenv(
    override=True,
)

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
    with open(
        f"{config.connected_dir}/preds/{config.pred_name}.pickle",
        "rb",
    ) as f:
        all_predictions = pickle.load(f)

    if config.is_preprocessed:
        data_encoder_path = (
            f"{config.custom_data_encoder_path}/{config.pretrained_model_name}"
        )
    else:
        data_encoder_path = config.pretrained_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        data_encoder_path,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoded_predictions = tokenizer.batch_decode(
        sequences=all_predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    generation_df = pd.read_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv"
    )
    original_generation_df = generation_df.copy()
    generation_df[config.target_column_name] = decoded_predictions
    generation_df[config.target_column_name] = generation_df[
        config.target_column_name
    ].str.replace(
        r"\s{2,}",
        " ",
        regex=True,
    )
    os.makedirs(
        f"{config.connected_dir}/from_decoded_predictions",
        exist_ok=True,
    )
    generation_df.to_csv(
        f"{config.connected_dir}/from_decoded_predictions/{config.pred_name}.csv",
        index=False,
    )
    saved_df = pd.read_csv(
        f"{config.connected_dir}/from_decoded_predictions/{config.pred_name}.csv"
    )
    if len(saved_df) > len(original_generation_df):
        original_values = set(original_generation_df["fname"].tolist())
        saved_values = set(saved_df["fname"].tolist())
        conditions_to_remove = list(saved_values - original_values)
        saved_df = saved_df[~saved_df["fname"].isin(conditions_to_remove)]
    saved_df.to_csv(
        f"{config.connected_dir}/from_decoded_predictions/{config.pred_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    decode_predictions()
