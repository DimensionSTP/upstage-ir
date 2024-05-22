import os
import pickle
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
)
def merge_predictions(
    config: DictConfig,
) -> None:
    logits = []
    for per_device_file_name in os.listdir(f"{config.per_device_save_path}/logits"):
        if per_device_file_name.endswith(".npy"):
            per_device_logit = np.load(
                f"{config.per_device_save_path}/logits/{per_device_file_name}"
            )
            logits.append(per_device_logit)

    generation_dfs = []
    for per_device_file_name in os.listdir(
        f"{config.per_device_save_path}/generations"
    ):
        if per_device_file_name.endswith(".csv"):
            per_device_generation_df = pd.read_csv(
                f"{config.per_device_save_path}/generations/{per_device_file_name}"
            )
            per_device_generation_df.fillna("_")
            generation_dfs.append(per_device_generation_df)

    all_logits = np.concatenate(
        logits,
        axis=0,
    )
    unique_indices = np.unique(all_logits[:, :, -1])
    unique_indices = unique_indices.astype(int)
    unique_all_logits = all_logits[unique_indices]
    sorted_logits_with_indices = unique_all_logits[
        np.argsort(unique_all_logits[:, 0, -1])
    ]
    generation_df = pd.read_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv"
    )
    sorted_logits = sorted_logits_with_indices[: len(generation_df), :, :-1]
    all_predictions = np.argmax(
        sorted_logits,
        axis=-1,
    )
    if not os.path.exists(f"{config.connected_dir}/logits"):
        os.makedirs(
            f"{config.connected_dir}/logits",
            exist_ok=True,
        )
    with open(
        f"{config.connected_dir}/logits/{config.logit_name}.pickle",
        "wb",
    ) as f:
        pickle.dump(
            sorted_logits,
            f,
        )
    if not os.path.exists(f"{config.connected_dir}/preds"):
        os.makedirs(
            f"{config.connected_dir}/preds",
            exist_ok=True,
        )
    with open(
        f"{config.connected_dir}/preds/{config.pred_name}.pickle",
        "wb",
    ) as f:
        pickle.dump(
            all_predictions,
            f,
        )

    combined_generation_df = pd.concat(generation_dfs)
    sorted_generation_df = combined_generation_df.sort_values(by="index").reset_index()
    all_generations = sorted_generation_df[config.target_column_name]
    if len(all_generations) < len(generation_df):
        raise ValueError(
            f"Length of all_generations {len(all_generations)} is shorter than length of predict data {len(generation_df)}."
        )
    if len(all_generations) > len(generation_df):
        all_generations = all_generations[: len(generation_df)]
    generation_df[config.target_column_name] = all_generations
    if not os.path.exists(f"{config.connected_dir}/submissions"):
        os.makedirs(
            f"{config.connected_dir}/submissions",
            exist_ok=True,
        )
    generation_df.to_csv(
        f"{config.connected_dir}/submissions/{config.submission_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    merge_predictions()
