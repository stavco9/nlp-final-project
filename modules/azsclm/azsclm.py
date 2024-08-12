import torch
import torch.nn as nn
from modules.azsclm.azsc_transformer import Azsc_Transformers

class Config_transformer:
    def __init__(self, src_vocab_size=5000, tgt_size=500, src_pad_idx=0, trg_pad_idx=0, embed_size=256, num_layers=3, forward_expansion=1024, num_heads=4, dropout=0.1, device="cuda", max_length=100):
        self.src_vocab_size = src_vocab_size
        self.tgt_size = tgt_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.max_length = max_length
    
class AZSC_LanguageModel(nn.Module):
    def __init__(self, tokenizer, text_length):
        super(AZSC_LanguageModel, self).__init__()
        vocab_size = tokenizer.vocab_size
        config = Config_transformer(src_vocab_size=vocab_size, tgt_size=100, num_layers=6, max_length=text_length)

        self.tokenizer = tokenizer
        self.transformers_chain = Azsc_Transformers(config)
        self.fc_out = nn.Linear(config.tgt_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, src):
        # get a tensor of shape (batch_size, seq_length, tgt_size)
        out = self.transformers_chain(src)
        # get a tensor of shape (batch_size, tgt_size)
        out = out.sum(dim=1)
        out = self.fc_out(out)
        out = self.tanh(out)
        return out