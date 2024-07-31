import dotenv

dotenv.load_dotenv(
    override=True,
)

from transformers import AutoTokenizer, AutoModel

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def merge_model(
    config: DictConfig,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}"
    )
    model = AutoModel.from_pretrained(config.pretrained_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(config.merged_model_path)


if __name__ == "__main__":
    merge_model()
