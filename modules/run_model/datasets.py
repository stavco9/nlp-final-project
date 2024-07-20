import torch
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, data):
        # need to assert that all d[0] are of the same length
        # d[0] is already tokenized so a tensor of type long
        self.data = [(d[0], torch.tensor(d[1], dtype=torch.float32)) for d in data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx][0]
        trg = self.data[idx][1]
        return src, trg

def homogonize_token_length(text, tokenizer, text_length):
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)['input_ids']
    if tokenized_text.shape[1] < text_length:
        tokenized_text = torch.cat((tokenized_text, torch.zeros(tokenized_text.shape[0], text_length - tokenized_text.shape[1])), dim=1)
    else:
        tokenized_text = tokenized_text[:, :text_length]
    return tokenized_text

class Datasets:
    def __init__(self, json_file, tokenizer, text_length, ratios=[0.8, 0.1, 0.1], batch_size=32):
        self.json_file = json_file
        assert sum(ratios) == 1
        self.ratios = ratios
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_length = text_length
        self.load_data()
    
    def load_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)

        random.shuffle(data)

        train_size = int(self.ratios[0] * len(data))
        val_size = int(self.ratios[1] * len(data))
        #test_size = len(data) - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        #tokenizer = lambda x: tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        train_data1 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[1])] for d in train_data]
        val_data1 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[1])] for d in val_data]
        test_data1 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[1])] for d in test_data]

        train_data2 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[2])] for d in train_data]
        val_data2 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[2])] for d in val_data]
        test_data2 = [[homogonize_token_length(d[0], self.tokenizer, self.text_length), float(d[2])] for d in test_data]

        train_dataset1 = CustomDataset(train_data1)
        val_dataset1 = CustomDataset(val_data1)
        test_dataset1 = CustomDataset(test_data1)

        train_dataset2 = CustomDataset(train_data2)
        val_dataset2 = CustomDataset(val_data2)
        test_dataset2 = CustomDataset(test_data2)

        # Create data loaders
        self.train_loader1 = DataLoader(train_dataset1, batch_size=self.batch_size, shuffle=True)
        self.val_loader1 = DataLoader(val_dataset1, batch_size=self.batch_size, shuffle=False)
        self.test_loader1 = DataLoader(test_dataset1, batch_size=self.batch_size, shuffle=False)

        self.train_loader2 = DataLoader(train_dataset2, batch_size=self.batch_size, shuffle=True)
        self.val_loader2 = DataLoader(val_dataset2, batch_size=self.batch_size, shuffle=False)
        self.test_loader2 = DataLoader(test_dataset2, batch_size=self.batch_size, shuffle=False)
        