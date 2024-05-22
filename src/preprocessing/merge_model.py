from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_model(
    path: str,
    pretrained_model_name: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{path}/merged_tokenizer/{pretrained_model_name}"
    )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(f"{path}/merged_model/{pretrained_model_name}")


if __name__ == "__main__":
    PATH = "/data/upstage-nlp/data"
    PRETRAINED_MODEL_NAME = "vicgalle/SOLAR-13B-Instruct-v1.0"
    merge_model(
        path=PATH,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
    )
