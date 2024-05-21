import os

import pandas as pd


def make_corpus(
    path: str,
) -> None:
    train_df = pd.read_csv(f"{path}/train.csv")
    val_df = pd.read_csv(f"{path}/dev.csv")
    test_df = pd.read_csv(f"{path}/test.csv")

    if not os.path.exists(f"{path}/corpus"):
        os.makedirs(
            f"{path}/corpus",
            exist_ok=True,
        )

    with open(f"{path}/corpus/corpus.txt", "w", encoding="utf-8") as f:
        for line in train_df["dialogue"]:
            f.write(line + "\n")
        for line in train_df["summary"]:
            f.write(line + "\n")
    with open(f"{path}/corpus/corpus.txt", "a", encoding="utf-8") as f:
        for line in val_df["dialogue"]:
            f.write(line + "\n")
        for line in val_df["summary"]:
            f.write(line + "\n")
    with open(f"{path}/corpus/corpus.txt", "a", encoding="utf-8") as f:
        for line in test_df["dialogue"]:
            f.write(line + "\n")


if __name__ == "__main__":
    PATH = "/data/upstage-nlp/data"
    make_corpus(
        path=PATH,
    )
