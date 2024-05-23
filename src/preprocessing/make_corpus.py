import os

import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def make_corpus(
    config: DictConfig,
) -> None:
    train_df = pd.read_csv(f"{config.connected_dir}/data/train.csv")
    val_df = pd.read_csv(f"{config.connected_dir}/data/dev.csv")
    test_df = pd.read_csv(f"{config.connected_dir}/data/test.csv")

    if not os.path.exists(f"{config.connected_dir}/data/corpus"):
        os.makedirs(
            f"{config.connected_dir}/data/corpus",
            exist_ok=True,
        )

    with open(
        f"{config.connected_dir}/data/corpus/corpus.txt", "w", encoding="utf-8"
    ) as f:
        for line in train_df["dialogue"]:
            f.write(line + "\n")
        for line in train_df["summary"]:
            f.write(line + "\n")
    with open(
        f"{config.connected_dir}/data/corpus/corpus.txt", "a", encoding="utf-8"
    ) as f:
        for line in val_df["dialogue"]:
            f.write(line + "\n")
        for line in val_df["summary"]:
            f.write(line + "\n")
    with open(
        f"{config.connected_dir}/data/corpus/corpus.txt", "a", encoding="utf-8"
    ) as f:
        for line in test_df["dialogue"]:
            f.write(line + "\n")


if __name__ == "__main__":
    make_corpus()
