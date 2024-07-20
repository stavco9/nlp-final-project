
import torch
import torch.nn as nn
import torch.optim as optim
from modules.azsclm.azsclm import AZSC_LanguageModel
from modules.run_model.train import Train, Config_train
from modules.run_model.datasets import Datasets
from transformers import AutoTokenizer

def run():
    text_length = 8192
    tokenizer_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    datasets = Datasets("data/data/vids_sentiment.json", tokenizer, text_length)

    # Instantiate the model
    model = AZSC_LanguageModel(tokenizer, text_length).to("cuda")

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Instantiate the loss function
    criterion = nn.MSELoss()

    traind_data = datasets.train_loader1
    config = Config_train(num_epochs=10, train_data=traind_data, model=model, optimizer=optimizer, criterion=criterion, device="cuda")
    
    # Instantiate the Train object
    trainer = Train(config, data_name='classical', print_level=0)

    # Start training
    trainer.train()

run()