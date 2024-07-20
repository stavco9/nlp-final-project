import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter object
log_dir = "data/runs/tensorboard_visualization"

class Board:
    def __init__(self, info, print_level=0):
        self.print_level = print_level # 0 - strictly necessary prints, 1 - important prints, 2 - prints, 3 - debug print
        self.writer = SummaryWriter(log_dir)
        self.epoch = 0

    def info_handler(self, loss, batch, lenX, size, epoch_loss, name, losses = None):
        epoch_loss += loss
        if batch % 100 != 0:
            pass
        # write to tensorboard the overall loss
        self.writer.add_scalar(name,
                        epoch_loss,
                        self.epoch * lenX + batch)
        if losses != None:
            for section_name, section_loss in losses.items():
                self.writer.add_scalar(name + '_' + section_name, section_loss, self.epoch * lenX + batch)
        return epoch_loss

    def close(self):
        self.writer.close()
