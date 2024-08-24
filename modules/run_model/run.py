
import torch
import torch.nn as nn
import torch.optim as optim
from modules.azsclm.azsclm import AZSC_LanguageModel
from modules.run_model.train import Model, Config_Model
from modules.run_model.datasets import Datasets
from transformers import AutoTokenizer
from transformers import LongformerModel, LongformerTokenizer
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

def run(FromScratch = True, do_GPT = True, do_BERT = True):
    torch.cuda.empty_cache()

    print(torch.cuda.memory_summary(device="cuda", abbreviated=True))
    if FromScratch:
        print("From Scratch")
        TEXT_LENGTH = 4096
        tokenizer_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        datasets = Datasets("data/sentiment/vids_sentiment_fixed_new.json", tokenizer, TEXT_LENGTH, ratios=[0.8, 0.1, 0.1], batch_size=1, device="cuda")
        # Instantiate the model
        model = AZSC_LanguageModel(tokenizer, TEXT_LENGTH).to("cuda")
        weights_path = "data/sentiment/weights_from_scratch_"
        isLongformer = False

    else:
        print("From Pretrained")
        TEXT_LENGTH = 4096
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-' + str(TEXT_LENGTH))
        datasets = Datasets("data/sentiment/vids_sentiment_fixed_new.json", tokenizer, TEXT_LENGTH, ratios=[0.8, 0.1, 0.1], batch_size=1, device="cuda")
        model = LongformerModel.from_pretrained('allenai/longformer-base-' + str(TEXT_LENGTH)).to("cuda")
        weights_path = "data/sentiment/weights_from_pretrained_"
        isLongformer = True

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Instantiate the loss function
    criterion = nn.MSELoss()

    if do_GPT:
        # the gpt model
        traind_data_gpt = datasets.train_loader1
        valid_data_gpt = datasets.val_loader1
        test_data_gpt = datasets.test_loader1

        config_gpt = Config_Model(num_epochs=25, train_data=traind_data_gpt, valid_data=valid_data_gpt, test_data=test_data_gpt, model=model, optimizer=optimizer, threshold_percentage = 5, criterion=criterion, device="cuda")
        model_gpt = Model(config_gpt, data_name='GPT', isLongformer=isLongformer)

        model_gpt.train()
        model_gpt.test()
        precision, recall, f1 = model_classical.calculate_metrics()
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        torch.save(model.state_dict(), weights_path + 'gpt.pth')
        
    # reset the model for the second type of learning
    if FromScratch:
        model = AZSC_LanguageModel(tokenizer, TEXT_LENGTH).to("cuda")
    else:
        model = LongformerModel.from_pretrained('allenai/longformer-base-' + str(TEXT_LENGTH)).to("cuda")

    if do_BERT:
        trained_data_classical = datasets.train_loader2
        valid_data_classical = datasets.val_loader2
        test_data_classical = datasets.test_loader2
        config_classical = Config_Model(num_epochs=25, train_data=trained_data_classical, valid_data=valid_data_classical, test_data=test_data_classical, model=model, optimizer=optimizer, threshold_percentage = 5, criterion=criterion, device="cuda")
        model_classical = Model(config_classical, data_name='classical', isLongformer=isLongformer)
            
        model_classical.train()
        model_classical.test()
        precision, recall, f1 = model_classical.calculate_metrics()

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        torch.save(model.state_dict(), weights_path + 'classical.pth')

run(True, False, True)
#python3 -m modules.run_model.run