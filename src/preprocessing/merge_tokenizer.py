import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

from transformers import AutoTokenizer
import sentencepiece as spm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def merge_tokenizer(
    config: DictConfig,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.connected_dir}/data/sentencepiece/{config.dataset_name}.model")

    new_tokens = []
    for idx in range(sp.get_piece_size()):
        token = sp.id_to_piece(idx)
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

    tokenizer.add_tokens(new_tokens)

    if not os.path.exists(config.custom_data_encoder_path):
        os.makedirs(
            config.custom_data_encoder_path,
            exist_ok=True,
        )
    tokenizer.save_pretrained(config.custom_data_encoder_path)


if __name__ == "__main__":
    merge_tokenizer()
