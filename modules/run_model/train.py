
import torch
import torch.nn as nn
import json
from modules.azsclm.board import Board
from torch.utils.data import DataLoader


class Config_Model:
    def __init__(self, num_epochs, train_data, valid_data, test_data, model, optimizer, threshold_percentage, criterion, device):
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.threshold_percentage = threshold_percentage


class Model:
    def __init__(self, config, data_name = 'GPT'):
        self.num_epochs = config.num_epochs
        self.train_data = config.train_data
        self.valid_data = config.valid_data
        self.test_data = config.test_data
        self.pred_score = []
        self.true_score = []
        self.device = config.device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.data_name = 'comments_controversy_' + data_name
        self.max_available = 1
        self.min_available = -1
        self.board = Board()

    def calculate_accuracy(self, min_item, max_item, pred, true):
        pred = max(pred, self.min_available)
        pred = min(pred, self.max_available)
        true = max(true, self.min_available)
        true = min(true, self.max_available)

        acc = 1 - (abs(pred - true) * 100 / abs(max_item - min_item))
        return float(acc)

    def train(self):
        size = len(self.train_data.dataset)
        num_train_batches = len(self.train_data)
        num_valid_batches = len(self.valid_data)
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_loss = 0

            for batch_idx, (src, trg) in enumerate(self.train_data):
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src)

                self.optimizer.zero_grad()
                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                train_loss = loss.item()
                self.optimizer.step()
                epoch_loss = self.board.info_handler(loss=train_loss, batch=batch_idx, size=size, epoch_loss=epoch_loss, name='(training) ' + self.data_name)

            print("Evaluating with validation set")
            self.model.eval()
            running_valid_loss = 0
            total_acc = 0

            list_targets = [item[1].to('cuda') for item in self.valid_data]
            list_targets = torch.stack(list_targets, dim=1).to('cuda')
            max_item = min(torch.max(list_targets).item(), self.max_available)
            min_item = max(torch.min(list_targets).item(), self.min_available)

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for batch_idx, (src, trg) in enumerate(self.valid_data):
                    src, trg = src.to(self.device), trg.to(self.device)
                    output = self.model(src)
                    loss = self.criterion(output, trg)
                    valid_loss = loss.item()
                    running_valid_loss += valid_loss
                    total_acc += self.calculate_accuracy(max_item, min_item, output, trg)
                    avg_acc = total_acc / (batch_idx + 1)
                    print(f"Accuracy: {avg_acc}")

            avg_train_loss = epoch_loss / num_train_batches
            avg_valid_loss = running_valid_loss / num_valid_batches
            print(f"Average train loss: {avg_train_loss}, avarage valid loss: {avg_valid_loss}")
            avg_acc = total_acc / num_valid_batches
            print(f"Average accuracy rate: {avg_acc}")

        batch = next(iter(self.train_data))
        src = batch[0].to(self.device)
        self.board.add_graph(self.model, src)

        # save the model
        torch.save(self.model.state_dict(), 'data/models/' + self.data_name + '.pth')

    def test(self):
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)
        self.model.eval()
        epoch_loss = 0.0
        sections_losses = {}
        total_acc = 0

        list_targets = [item[1].to('cuda') for item in self.test_data]
        list_targets = torch.stack(list_targets, dim=1).to('cuda')
        max_item = min(torch.max(list_targets).item(), self.max_available)
        min_item = max(torch.min(list_targets).item(), self.min_available)
        with torch.no_grad():
            for batch_idx, (src, trg) in enumerate(self.test_data):
                src, trg = src.to(self.device), trg.to(self.device)
                pred = self.model(src)
                pred = pred.to(self.device)
                loss = self.criterion(pred, trg)
                #for section_name, section_loss in loss.last_losses.items():
                #    if section_name not in sections_losses:
                #        sections_losses[section_name] = torch.Tensor([0]).to(self.device)
                #    sections_losses[section_name] += section_loss
                #for section_name in sections_losses.keys():
                #    sections_losses[section_name] /= size
                epoch_loss = self.board.info_handler(loss=loss.item(), batch=batch_idx, size=size, epoch_loss=epoch_loss, name = '(test) ' + self.data_name)
                avg_loss = epoch_loss / num_batches
                print(f"Average loss: {avg_loss}")
                total_acc += self.calculate_accuracy(max_item, min_item, pred, trg)
                avg_acc = total_acc / (batch_idx + 1)
                print(f"Accuracy: {avg_acc}")
                self.pred_score.extend(pred.cpu().numpy())
                self.true_score.extend(trg.cpu().numpy())

        avg_loss = epoch_loss / num_batches
        print(f"Average loss rate: {avg_acc}")

        avg_acc = total_acc / num_batches
        print(f"Average accuracy rate: {avg_acc}")

    def calculate_metrics(self):
      # Convert lists to tensors for calculation
      true_score_tensor = torch.tensor(self.true_score)
      pred_score_tensor = torch.tensor(self.pred_score)

      # Calculating precision, recall, and F1 score using PyTorch
      TP = ((pred_score_tensor >= 0) & (true_score_tensor >= 0)).sum().item()
      FP = ((pred_score_tensor >= 0) & (true_score_tensor < 0)).sum().item()
      FN = ((pred_score_tensor < 0) & (true_score_tensor >= 0)).sum().item()

      precision = TP / (TP + FP) if TP + FP > 0 else 0
      recall = TP / (TP + FN) if TP + FN > 0 else 0
      f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

      print(f'Precision: {precision}')
      print(f'Recall: {recall}')
      print(f'F1 Score: {f1}')

      return (precision, recall, f1)
