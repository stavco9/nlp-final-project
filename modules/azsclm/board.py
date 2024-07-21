import numpy as np
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=data/runs
# http://localhost:6007/

# Create a SummaryWriter object
log_dir = "data/runs/tensorboard_visualization"

class Board:
    def __init__(self):
        self.writer = SummaryWriter(log_dir)
        self.epoch = 0
        self.total_loss = 0

    def info_handler(self, loss, batch, size, epoch_loss, name):
        epoch_loss += loss
        self.total_loss += loss
        self.epoch += 1
        # write to tensorboard the overall loss
        self.writer.add_scalar(name + ' current loss: ', loss, self.epoch * size + batch)
        self.writer.add_scalar(name + ' total loss: ', self.total_loss, self.epoch)
        self.writer.add_scalar(name + ' epoch loss: ', epoch_loss, self.epoch * size + batch)
        return epoch_loss

    def close(self):
        self.writer.close()
