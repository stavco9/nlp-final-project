
import torch
import torch.nn as nn
import torch.optim as optim
from ..azsclm.azsclm import AZSC_LanguageModel
from transformers import AutoTokenizer
from train import Train, Config_train
from datasets import Datasets

def run():
    datasets = Datasets(json_file="data/data/vids_sentiment.json")

    tokenizer_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = lambda x: tokenizer(x, return_tensors="pt", truncation=True, padding=True)

    # Instantiate the model
    model = AZSC_LanguageModel(tokenizer).to("cuda")

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Instantiate the loss function
    criterion = nn.MSELoss()

    config = Config_train(num_epochs=10, train_data=None, model=model, optimizer=optimizer, criterion=criterion, device="cuda")

    traind_data = datasets.train_loader1
    # Instantiate the Train object
    trainer = Train(config, traind_data, data_name='classical', print_level=0)

    # Start training
    trainer.train()