
import torch
import torch.nn as nn
import json
from modules.azsclm.board import Board
from torch.utils.data import DataLoader


class Config_train:
    def __init__(self, num_epochs, train_data, model, optimizer, criterion, device):
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


class Train:
    def __init__(self, config, data_name = 'GPT'):
        self.num_epochs = config.num_epochs
        self.train_data = config.train_data
        self.device = config.device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.data_name = 'comments_controversy_' + data_name
        self.dataloader = config.train_data
        self.board = Board()

    def train(self):
        size = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_loss = 0
            for batch_idx, (src, trg) in enumerate(self.dataloader):
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src)

                self.optimizer.zero_grad()
                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.7)
                curr_loss = loss.item()
                self.optimizer.step()

            epoch_loss = self.board.info_handler(loss=curr_loss, batch=batch_idx, size=size, epoch_loss=epoch_loss, name='(training) ' + self.data_name)
            avg_loss = epoch_loss / num_batches
            print(f"Average loss: {avg_loss}")
        # save the model
        torch.save(self.model.state_dict(), 'data/models/' + self.data_name + '.pth')
