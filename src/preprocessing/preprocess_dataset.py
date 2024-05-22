import pandas as pd

from transformers import AutoTokenizer


def preprocess_dataset(
    dataset_path: str,
    dataset_file: str,
    tokenizer_path: str,
    max_length: int,
    save_prefix: str,
) -> None:
    df = pd.read_csv(f"{dataset_path}/{dataset_file}.csv")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = "너의 역할은 대화 내용을 요약해주는 요약 전문가야. 다음 사람들의 대화 내용을 보고 적절히 요약해줘."
        prompt = f"""### Instruction:
        {default_system_prompt} 
        ### Input:
        {data.strip()}
        ### Response:
        """.strip()
        return prompt

    df["prompt"] = df["dialogue"].apply(generate_prompt)

    def cut_prompt_to_length(
        prompt: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> str:
        tokens = tokenizer.tokenize(prompt)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        cut_prompt = tokenizer.convert_tokens_to_string(tokens)
        return cut_prompt

    df["cut_prompt"] = df["prompt"].apply(
        lambda x: cut_prompt_to_length(
            prompt=x,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    )
    df.to_csv(
        f"{dataset_path}/{save_prefix}_{dataset_file}.csv",
        index=False,
    )


if __name__ == "__main__":
    DATASET_PATH = "/data/upstage-nlp/data"
    DATASET_FILES = [
        "train",
        "dev",
        "test",
    ]
    TOKENIZER_PATH = (
        "/data/upstage-nlp/data/merged_tokenizer/vicgalle/SOLAR-13B-Instruct-v1.0"
    )
    MAX_LENGTH = 1024
    SAVE_PREFIX = "preprocessed"

    for dataset_file in DATASET_FILES:
        preprocess_dataset(
            dataset_path=DATASET_PATH,
            dataset_file=dataset_file,
            tokenizer_path=TOKENIZER_PATH,
            max_length=MAX_LENGTH,
            save_prefix=SAVE_PREFIX,
        )
