import os

import sentencepiece as spm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def train_tokenizer(
    config: DictConfig,
) -> None:
    if not os.path.exists(f"{config.connected_dir}/data/sentencepiece"):
        os.makedirs(
            f"{config.connected_dir}/data/sentencepiece",
            exist_ok=True,
        )

    spm.SentencePieceTrainer.train(
        input=f"{config.connected_dir}/data/corpus/corpus.txt",
        model_prefix=f"{config.connected_dir}/data/sentencepiece/dialogsum",
        vocab_size=1128,
    )


if __name__ == "__main__":
    train_tokenizer()
