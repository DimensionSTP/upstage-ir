from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class UpStageDocumentQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        is_preprocessed: bool,
        data_column_name: str,
        prompt_column_name: str,
        target_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        data_max_length: int,
        target_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.is_preprocessed = is_preprocessed
        self.data_column_name = data_column_name
        self.prompt_column_name = prompt_column_name
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.pretrained_model_name = pretrained_model_name
        if self.is_preprocessed:
            data_encoder_path = (
                f"{custom_data_encoder_path}/{self.pretrained_model_name}"
            )
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        if self.is_preprocessed:
            prompt = self.datas[idx] + self.labels[idx]
        else:
            prompt = self.generate_prompt(
                data=self.datas[idx],
                label=self.labels[idx],
            )
        encoded = self.encode_text(
            data=prompt,
            data_type="data",
        )
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/train.csv"
            else:
                csv_path = f"{self.data_path}/train.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/{self.split}.csv"
            else:
                csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        elif self.split == "predict":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/test.csv"
            else:
                csv_path = f"{self.data_path}/test.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        if self.is_preprocessed:
            datas = data[self.prompt_column_name].tolist()
        else:
            datas = data[self.data_column_name].apply(lambda x: x.strip()).tolist()
        labels = data[self.target_column_name].tolist()
        return {
            "datas": datas,
            "labels": labels,
        }

    def encode_text(
        self,
        data: str,
        data_type: str,
    ) -> Dict[str, torch.Tensor]:
        if data_type == "data":
            if self.split == "predict":
                max_length = self.data_max_length
            else:
                max_length = self.data_max_length + self.target_max_length
        elif data_type == "target":
            max_length = self.target_max_length
        else:
            raise ValueError(f"Inavalid data_type: {data_type}")
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def generate_prompt(
        self,
        data: str,
        label: str,
    ) -> str:
        default_system_prompt = """
너는 과학 질문에 대한 답변을 제공하는 챗봇이야.
너의 역할은 사용자들이 과학적 주제에 대해 궁금해하는 질문에 명확하고 정확한 답변을 제공하는 거야.
"""
        if self.split == "predict":
            prompt = f"""### Instruction:
{default_system_prompt} 

### Input(질문):
{data.strip()}

### Response(답변):
""".strip()
        else:
            prompt = f"""### Instruction:
{default_system_prompt} 

### Input(질문):
{data.strip()}

### Response(답변):
{label} """.strip()
        return prompt
