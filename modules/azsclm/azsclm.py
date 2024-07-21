import torch
import torch.nn as nn
from modules.azsclm.azsc_transformer import Azsc_Transformer

class Config_transformer:
    def __init__(self, src_vocab_size=5000, tgt_vocab_size=5000, src_pad_idx=0, trg_pad_idx=0, embed_size=512, num_layers=3, forward_expansion=1024, num_heads=8, dropout=0.1, device="cuda", max_length=100):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
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
        config = Config_transformer(src_vocab_size=vocab_size, tgt_vocab_size=0, num_layers=8, max_length=text_length)

        self.tokenizer = tokenizer
        self.transformers = Azsc_Transformer(config)
        fc_input_dim = text_length * config.embed_size
        self.fc_out = nn.Linear(fc_input_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, src):
        out = self.transformers(src)
        out = out.reshape(out.shape[0], -1)
        out = self.fc_out(out)
        out = self.tanh(out)
        return out