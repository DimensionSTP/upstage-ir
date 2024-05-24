from typing import Dict, Any, List

import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class UpStageDialoguesDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        is_preprocessed: bool,
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
        self.is_preprocessed = is_preprocessed
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
        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            encoded = self.encode_text(
                data=self.datas[idx],
                data_type="data",
            )
            label = self.encode_text(
                data=self.labels[idx],
                data_type="target",
            )["input_ids"]
            encoded["labels"] = label
        else:
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
        if self.split in ["train", "test"]:
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/{self.split}.csv"
            else:
                csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        elif self.split == "val":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/dev.csv"
            else:
                csv_path = f"{self.data_path}/dev.csv"
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
        if "bart" in self.pretrained_model_name or "t5" in self.pretrained_model_name:
            datas = data["dialogue"].tolist()
        else:
            if self.is_preprocessed:
                datas = data["cut_prompt"].tolist()
            else:
                datas = data["dialogue"].tolist()
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
            if (
                "bart" in self.pretrained_model_name
                or "t5" in self.pretrained_model_name
            ):
                max_length = self.data_max_length
            else:
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
        default_system_prompt = "너의 역할은 대화 내용을 요약해주는 요약 전문가야. 다음 사람들의 대화 내용을 보고 적절히 요약해줘."
        if self.split == "predict":
            prompt = f"""### Instruction:
            {default_system_prompt} 

            ### Input:
            {data.strip()}

            ### Response:
            """.strip()
        else:
            prompt = f"""### Instruction:
            {default_system_prompt} 

            ### Input:
            {data.strip()}

            ### Response:
            {label} """.strip()
        return prompt
