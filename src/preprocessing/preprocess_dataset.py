import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_dataset(
    config: DictConfig,
) -> None:
    df = pd.read_csv(f"{config.connected_dir}/data/{config.mode}.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        config.custom_data_encoder_path,
        use_fast=True,
    )

    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = """
너의 역할은 과학 질문에 대한 답변을 제공하는 챗봇이야.
너의 구체적인 목표는 사용자들이 과학적 주제에 대해 궁금해하는 질문에 명확하고 정확한 답변을 제공하는 것입니다.
하지만 과학 관련 질문이 아닌 경우, "답변 불가능."으로 응답해야 해.
이 규칙을 엄격히 준수해줘.
아래의 예시를 참고해.

**예시 1**:
질문:
태양은 어떻게 에너지를 생성하나요?
응답:
태양은 핵융합 반응을 통해 에너지를 생성합니다.
태양의 중심부에서 수소 원자들이 고온과 고압 상태에서 헬륨으로 융합되면서 엄청난 양의 에너지가 방출됩니다.
이 에너지는 태양의 표면을 통해 빛과 열의 형태로 우주로 방출됩니다.

**예시 2**:
질문:
이탈리아의 수도는 어디인가요?
응답:
답변 불가능.

**예시 3**:
질문:
물의 분자는 어떤 구조를 가지고 있나요?
응답:
물 분자는 두 개의 수소 원자와 한 개의 산소 원자로 이루어져 있습니다.
이들은 공유 결합을 통해 결합하며, 물 분자는 V자 모양을 하고 있습니다.
산소 원자는 약간 음전하를 띠고 수소 원자는 약간 양전하를 띠어 물 분자는 극성을 가지게 됩니다.

**예시 4**:
질문:
바나나의 영어 단어는 무엇인가요?
응답:
답변 불가능.

이와 같이, 과학 관련 질문에만 상세히 응답하고, 과학 관련이 아닌 질문에는 "답변 불가능."으로 응답해줘.
"""
        prompt = f"""### Instruction:
{default_system_prompt} 

### Input:
{data.strip()}

### Response:
""".strip()
        return prompt

    df["prompt"] = df[config.data_column_name].apply(generate_prompt)
    df[config.data_column_name] = df[config.data_column_name].apply(lambda x: x.strip())

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

    df[config.prompt_column_name] = df["prompt"].apply(
        lambda x: cut_prompt_to_length(
            prompt=x,
            tokenizer=tokenizer,
            max_length=config.data_max_length,
        )
    )
    if not os.path.exists(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}"
    ):
        os.makedirs(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}",
            exist_ok=True,
        )
    df.to_csv(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}/{config.mode}.csv",
        index=False,
    )


if __name__ == "__main__":
    preprocess_dataset()
