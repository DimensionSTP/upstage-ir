import os

from transformers import AutoTokenizer
import sentencepiece as spm


def merge_tokenizer(
    pretrained_model_name: str,
    custom_tokenizer_file: str,
    merged_tokenizer_save_path: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    sp = spm.SentencePieceProcessor()
    sp.load(custom_tokenizer_file)

    new_tokens = []
    for idx in range(sp.get_piece_size()):
        token = sp.id_to_piece(idx)
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

    tokenizer.add_tokens(new_tokens)

    if not os.path.exists(merged_tokenizer_save_path):
        os.makedirs(
            merged_tokenizer_save_path,
            exist_ok=True,
        )
    tokenizer.save_pretrained(merged_tokenizer_save_path)


if __name__ == "__main__":
    PRETRAINED_MODEL_NAME = "vicgalle/SOLAR-13B-Instruct-v1.0"
    CUSTOM_TOKENIZER_FILE = "/data/upstage-nlp/data/sentencepiece/dialogsum.model"
    MERGED_TOKENIZER_SAVE_PATH = (
        f"/data/upstage-nlp/data/merged_tokenizer/{PRETRAINED_MODEL_NAME}"
    )
    merge_tokenizer(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        custom_tokenizer_file=CUSTOM_TOKENIZER_FILE,
        merged_tokenizer_save_path=MERGED_TOKENIZER_SAVE_PATH,
    )
