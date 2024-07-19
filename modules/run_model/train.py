
import torch
import torch.nn as nn
import json
from ..azsclm.board import Board
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
    def __init__(self, config, dataloader, data_name = 'classical', print_level = 0):
        self.num_epochs = config.num_epochs
        self.train_data = config.train_data
        self.device = config.device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.data_name = 'comments_controversy_' + data_name
        self.dataloader = dataloader
        self.board = Board(self, print_level)

    def train(self):
        size = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            total_loss = 0
            for batch_idx, (src, trg) in enumerate(self.dataloader):
                src, trg = src.to(self.device), trg.to(self.device)
                self.model.train()
                output = self.model(src)

                # Assuming trg contains the ground truth for the numerical result
                target = self.dataloader[1].to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            epoch_loss = self.board.info_handler(loss=loss.item(), batch=batch_idx, lenX=len(self.dataloader), size=len(self.dataloader), epoch_loss=epoch_loss, name=self.data_name)
