
import torch
import torch.nn as nn
import torch.optim as optim
from modules.azsclm.azsclm import AZSC_LanguageModel
from modules.run_model.train import Model, Config_Model
from modules.run_model.datasets import Datasets
from transformers import AutoTokenizer
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

def run():
    torch.cuda.empty_cache()

    print(torch.cuda.memory_summary(device="cuda", abbreviated=True))
    
    text_length = 4096
    tokenizer_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    datasets = Datasets("data/sentiment/vids_sentiment.json", tokenizer, text_length, ratios=[0.8, 0.1, 0.1], batch_size=1, device="cuda")

    # Instantiate the model
    model = AZSC_LanguageModel(tokenizer, text_length).to("cuda")

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.1)

    # Instantiate the loss function
    criterion = nn.MSELoss()

    # the gpt model
    traind_data_gpt = datasets.train_loader1
    valid_data_gpt = datasets.val_loader1
    test_data_gpt = datasets.test_loader1

    config_gpt = Config_Model(num_epochs=150, train_data=traind_data_gpt, valid_data=valid_data_gpt, test_data=test_data_gpt, model=model, optimizer=optimizer, threshold_percentage = 5, criterion=criterion, device="cuda")
    # Instantiate the Train object
    model_gpt = Model(config_gpt, data_name='GPT')

    # the bert model
    trained_data_classical = datasets.train_loader2
    valid_data_classical = datasets.val_loader2
    test_data_classical = datasets.test_loader2
    config_classical = Config_Model(num_epochs=100, train_data=trained_data_classical, valid_data=valid_data_classical, test_data=test_data_classical, model=model, optimizer=optimizer, threshold_percentage = 5, criterion=criterion, device="cuda")
    # Instantiate the Train object
    model_classical = Model(config_classical, data_name='classical')
    

    # Start training
    model_classical.train()

    # Run tests
    model_classical.test()
    precision, recall, f1 = model_classical.calculate_metrics()

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

run()
#python3 -m modules.run_model.run