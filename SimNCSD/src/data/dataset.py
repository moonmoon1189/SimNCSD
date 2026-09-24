import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json


class SimNCSDDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=128, is_contrastive=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.is_contrastive = is_contrastive
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _tokenize_source(self, text):
        # Truncate source post: first 64 and last 64 tokens if > 128
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length - 2:
            half = (self.max_length - 2) // 2
            tokens = tokens[:half] + tokens[-half:]

        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}

    def _tokenize_comment(self, text):
        # Truncate comment: first 128 tokens
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_enc = self._tokenize_source(item['source_text'])

        if self.is_contrastive:
            # For contrastive learning, we need positive and hard negative comments
            pos_enc = self._tokenize_comment(item['pos_comment'])
            neg_enc = self._tokenize_comment(item['neg_comment'])
            return {
                "source_input_ids": source_enc['input_ids'],
                "source_attention_mask": source_enc['attention_mask'],
                "pos_input_ids": pos_enc['input_ids'].squeeze(0),
                "pos_attention_mask": pos_enc['attention_mask'].squeeze(0),
                "neg_input_ids": neg_enc['input_ids'].squeeze(0),
                "neg_attention_mask": neg_enc['attention_mask'].squeeze(0),
            }
        else:
            # For classifier training, we need source, comment, and label
            comment_enc = self._tokenize_comment(item['comment_text'])
            return {
                "source_input_ids": source_enc['input_ids'],
                "source_attention_mask": source_enc['attention_mask'],
                "comment_input_ids": comment_enc['input_ids'].squeeze(0),
                "comment_attention_mask": comment_enc['attention_mask'].squeeze(0),
                "label": torch.tensor(item['label'], dtype=torch.float)
            }