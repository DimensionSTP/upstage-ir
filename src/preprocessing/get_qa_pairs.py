import dotenv

dotenv.load_dotenv(
    override=True,
)

from typing import Dict, List, Union
import os
import warnings

warnings.filterwarnings("ignore")

import json

import pandas as pd
from tqdm import tqdm

from openai import OpenAI
import google.generativeai as genai

import hydra
from omegaconf import DictConfig


def get_question_from_chatgpt(
    model_name: str,
    prompt: str,
) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "너는 답변을 보고 해당 답변이 챗봇으로부터 나올 수 있도록 질문을 유추해서 생성해주는 질문 유추 전문가야.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content


def get_question_from_gemini(
    model_name: str,
    prompt: str,
) -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def load_documents(
    documents_path: str,
) -> List[Dict[str, str]]:
    documents = []
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def create_qa_pairs(
    documents: List[Dict[str, str]],
    get_question_func: Union[get_question_from_chatgpt, get_question_from_gemini],
    model_name: str,
    output_file_name: str,
) -> None:
    qa_data = []
    for doc in tqdm(documents):
        answer = doc["content"]
        prompt = f"""{answer}라는 답변이 있다고 하자. 
해당 답변이 나오게 가장 좋은 한글 질문을 1개만 생성해서 해당 질문 문장만 답변해줘."""
        try:
            question = get_question_func(
                model_name=model_name,
                prompt=prompt,
            )
        except Exception as e:
            question = f"Error: {e}"
        qa_data.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    qa_df = pd.DataFrame(qa_data)
    qa_df.to_csv(
        output_file_name,
        index=False,
    )
    print(f"QA pairs saved to {output_file_name}")


def merge_and_clean_qa_pairs(
    data_path: str,
    chatgpt_model_names: List[str],
    gemini_model_names: List[str],
    output_file_name: str,
) -> None:
    model_names = chatgpt_model_names + gemini_model_names
    dataframes = [
        pd.read_csv(f"{data_path}/{model_name}.csv") for model_name in model_names
    ]
    merged_df = pd.concat(dataframes)
    merged_df = merged_df[
        ~merged_df["question"].str.contains(
            "Error",
            na=False,
        )
    ]
    merged_df = merged_df.drop_duplicates(subset="question")
    merged_df.to_csv(
        f"{data_path}/{output_file_name}.csv",
        index=False,
    )


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def get_qa_pairs(
    config: DictConfig,
) -> None:
    data_path = f"{config.connected_dir}/data"
    document_path = f"{data_path}/documents.jsonl"
    openai_api_key = config.openai_api_key
    google_api_key = config.google_api_key
    chatgpt_model_names = [
        "gpt-3.5-turbo",
        "gpt-4o",
    ]
    gemini_model_names = [
        "gemini-1.0-pro-latest",
        "gemini-1.5-flash-latest",
    ]

    os.environ["OPENAI_API_KEY"] = openai_api_key
    genai.configure(api_key=google_api_key)
    documents = load_documents(document_path)

    for i in range(2):
        create_qa_pairs(
            documents=documents,
            get_question_func=get_question_from_chatgpt,
            model_name=chatgpt_model_names[i],
            output_file_name=f"{data_path}/{chatgpt_model_names[i]}.csv",
        )
        create_qa_pairs(
            documents=documents,
            get_question_func=get_question_from_gemini,
            model_name=gemini_model_names[i],
            output_file_name=f"{data_path}/{gemini_model_names[i]}.csv",
        )

    merge_and_clean_qa_pairs(
        data_path=data_path,
        chatgpt_model_names=chatgpt_model_names,
        gemini_model_names=gemini_model_names,
        output_path="merged_cleaned_file",
    )


if __name__ == "__main__":
    get_qa_pairs()
