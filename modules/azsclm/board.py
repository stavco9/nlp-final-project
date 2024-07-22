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

    def info_handler(self, loss, batch, size, epoch_loss, name):
        epoch_loss += loss
        self.epoch += 1
        # write to tensorboard the overall loss
        self.writer.add_scalar(name + ' current loss: ', loss, self.epoch * size + batch)
        if batch == size - 1:
            self.writer.add_scalar(name + ' epoch loss: ', epoch_loss, self.epoch)
        return epoch_loss

    def add_graph(self, model, input):
        self.writer.add_graph(model, input)

    def close(self):
        self.writer.close()
