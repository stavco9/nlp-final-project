import torch
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader
import random

def custom_collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.stack(src).long()  # Ensure src is Long
    src = src.view(src.size(0), -1)
    tgt = torch.stack(tgt).float()  # Ensure tgt is Float
    tgt = tgt.view(tgt.size(0), -1)
    return src, tgt

class CustomDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = self.texts[idx]
        trg = self.scores[idx]
        return src, trg

def homogonize_token_length(text, tokenizer, text_length, device="cuda"):
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)['input_ids'].long().to(device)
    if tokenized_text.shape[1] < text_length:
        tokenized_text = torch.cat((tokenized_text, torch.zeros(tokenized_text.shape[0], text_length - tokenized_text.shape[1]).to(device)), dim=1)
    else:
        tokenized_text = tokenized_text[:, :text_length]
    return tokenized_text

class Datasets:
    def __init__(self, json_file, tokenizer, text_length, ratios=[0.8, 0.1, 0.1], batch_size=1, device="cuda"):
        self.json_file = json_file
        assert sum(ratios) == 1
        self.ratios = ratios
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_length = text_length
        self.device = device
        self.load_data()

    def load_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)

        random.shuffle(data)
        tokens = [homogonize_token_length(d[0], self.tokenizer, self.text_length, self.device) for d in data]
        GPT_scores = [torch.tensor(d[1], dtype=torch.float32).to(self.device) for d in data]
        BERT_scores = [torch.tensor(d[2], dtype=torch.float32).to(self.device) for d in data]

        train_size = int(self.ratios[0] * len(data))
        val_size = int(self.ratios[1] * len(data))
        #test_size = len(data) - train_size - val_size

        train_text = tokens[:train_size]
        val_text = tokens[train_size:train_size + val_size]
        test_text = tokens[train_size + val_size:]

        train_scores1 = GPT_scores[:train_size]
        val_scores1 = GPT_scores[train_size:train_size + val_size]
        test_scores1 = GPT_scores[train_size + val_size:]

        train_scores2 = BERT_scores[:train_size]
        val_scores2 = BERT_scores[train_size:train_size + val_size]
        test_scores2 = BERT_scores[train_size + val_size:]

        train_dataset1 = CustomDataset(train_text, train_scores1)
        val_dataset1 = CustomDataset(val_text, val_scores1)
        test_dataset1 = CustomDataset(test_text, test_scores1)

        train_dataset2 = CustomDataset(train_text, train_scores2)
        val_dataset2 = CustomDataset(val_text, val_scores2)
        test_dataset2 = CustomDataset(test_text, test_scores2)

        # Create data loaders
        self.train_loader1 = DataLoader(train_dataset1, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader1 = DataLoader(val_dataset1, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.test_loader1 = DataLoader(test_dataset1, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        self.train_loader2 = DataLoader(train_dataset2, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader2 = DataLoader(val_dataset2, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.test_loader2 = DataLoader(test_dataset2, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
