# create a DataLoader objects for training, testing and validation data
# use vids_sentiment.json. the first element of the tuple is the transcript and the second or third element is the score (second element is comments_controversy_GPT and third element is comments_controversy_classical)
# create different data loader for different models

import numpy as np
import json
from torch.utils.data import Dataset,DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, data, score_method):
        self.data = data
        self.score_method = score_method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][0]
        score = self.data[idx][self.score_method]
        return text, score

class Datasets:
    def __init__(self, json_file, ratios=[0.8, 0.1, 0.1], batch_size=32):
        self.transcripts = []
        self.scores = []
        self.json_file = json_file
        assert sum(ratios) == 1
        self.ratios = ratios
        self.batch_size = batch_size
        self.load_data()
    
    def load_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)

        random.shuffle(data)

        train_size = int(self.ratios[0] * len(data))
        val_size = int(self.ratios[1] * len(data))
        test_size = len(data) - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]

        train_dataset1 = CustomDataset(train_data, score_method=1)
        val_dataset1 = CustomDataset(val_data, score_method=1)
        test_dataset1 = CustomDataset(test_data, score_method=1)

        train_dataset2 = CustomDataset(train_data, score_method=2)
        val_dataset2 = CustomDataset(val_data, score_method=2)
        test_dataset2 = CustomDataset(test_data, score_method=2)

        # Create data loaders
        self.train_loader1 = DataLoader(train_dataset1, batch_size=self.batch_size, shuffle=True)
        self.val_loader1 = DataLoader(val_dataset1, batch_size=self.batch_size, shuffle=False)
        self.test_loader1 = DataLoader(test_dataset1, batch_size=self.batch_size, shuffle=False)

        self.train_loader2 = DataLoader(train_dataset2, batch_size=self.batch_size, shuffle=True)
        self.val_loader2 = DataLoader(val_dataset2, batch_size=self.batch_size, shuffle=False)
        self.test_loader2 = DataLoader(test_dataset2, batch_size=self.batch_size, shuffle=False)
        