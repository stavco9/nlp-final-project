
import torch
import torch.nn as nn
import torch.optim as optim
from modules.azsclm.azsclm import AZSC_LanguageModel
from modules.run_model.train import Train, Config_train
from modules.run_model.datasets import Datasets
from transformers import AutoTokenizer
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

def run():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device="cuda", abbreviated=True))

    text_length = 4096#8192
    tokenizer_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    datasets = Datasets("data/data/vids_sentiment.json", tokenizer, text_length, ratios=[0.8, 0.1, 0.1], batch_size=2, device="cuda")

    # Instantiate the model
    model = AZSC_LanguageModel(tokenizer, text_length).to("cuda")

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)

    # Instantiate the loss function
    criterion = nn.MSELoss()

    # the gpt model
#    traind_data = datasets.train_loader1
#    config = Config_train(num_epochs=150, train_data=traind_data, model=model, optimizer=optimizer, criterion=criterion, device="cuda")
    # Instantiate the Train object
#    trainer = Train(config, data_name='GPT')

    # the bert model
    traind_data = datasets.train_loader2
    config = Config_train(num_epochs=100, train_data=traind_data, model=model, optimizer=optimizer, criterion=criterion, device="cuda")
    # Instantiate the Train object
    trainer = Train(config, data_name='classical')
    

    # Start training
    trainer.train()

run()
#python3 -m modules.run_model.run