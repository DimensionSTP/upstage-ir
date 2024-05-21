import os

import sentencepiece as spm


def train_tokenizer(
    input_file: str,
    output_path: str,
    tokenizer_name: str,
    vocab_size: int,
) -> None:
    if not os.path.exists(output_path):
        os.makedirs(
            output_path,
            exist_ok=True,
        )

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=f"{output_path}/{tokenizer_name}",
        vocab_size=vocab_size,
    )


if __name__ == "__main__":
    INPUT_FILE = "/data/upstage-nlp/data/corpus/corpus.txt"
    OUTPUT_PATH = "/data/upstage-nlp/data/sentencepiece"
    TOKENIZER_NAME = "dialogsum"
    VOCAB_SIZE = 1128
    train_tokenizer(
        input_file=INPUT_FILE,
        output_path=OUTPUT_PATH,
        tokenizer_name=TOKENIZER_NAME,
        vocab_size=VOCAB_SIZE,
    )
