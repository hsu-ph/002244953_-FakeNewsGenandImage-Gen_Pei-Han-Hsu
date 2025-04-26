import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class FakeRealNewsDataset(Dataset):
    def __init__(self, fake_path, real_path, tokenizer_name="gpt2", max_length=512):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and label data
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        fake_df["label"] = "<FAKE>"
        real_df["label"] = "<REAL>"
        df = pd.concat([fake_df, real_df], ignore_index=True)
        df.dropna(subset=["title", "text"], inplace=True)
        df.drop_duplicates(subset=["text"], inplace=True)
        self.data = df.reset_index(drop=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"{row['label']} {row['title']}"
        target_text = row['text']

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
# dataset_a = FakeRealNewsDataset("Fake.csv", "True.csv")
