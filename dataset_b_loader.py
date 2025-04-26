import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class CNNDailyMailDataset(Dataset):
    def __init__(self, csv_paths, tokenizer_name="gpt2", max_length=512):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and combine multiple CSV files
        dfs = [pd.read_csv(path) for path in csv_paths]
        df = pd.concat(dfs, ignore_index=True)
        df.dropna(subset=["article", "highlights"], inplace=True)
        df["source_style"] = df["id"].apply(lambda x: "<US>" if "cnn" in x.lower() else "<UK>")
        self.data = df.reset_index(drop=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"{row['source_style']} {row['highlights']}"
        target_text = row['article']

        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

# Usage:
# dataset_b = CNNDailyMailDataset(["test.csv", "validation.csv"])
