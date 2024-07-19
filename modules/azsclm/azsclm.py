import torch
import torch.nn as nn
from azsc_transformer import Azsc_Transformer

class Config_part:
    def __init__(self):
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.embed_size = 256
        self.num_layers = 3
        self.forward_expansion = 4
        self.heads = 8
        self.dropout = 0.1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 100

class AZSC_LanguageModel(nn.Module):
    def __init__(self, tokenizer):
        super(AZSC_LanguageModel, self).__init__()

        config_one = Config_part()
        config_one.max_length = 8192
        config_one.num_layers = 2
        config_two = Config_part()
        config_two.num_layers = 6
        configs = [config_one, config_two]

        self.tokenizer = tokenizer
        self.transformers = [Azsc_Transformer(
            src_pad_idx=config.src_pad_idx,
            trg_pad_idx=config.trg_pad_idx,
            embed_size=config.embed_size,
            num_layers=config.num_layers,
            forward_expansion=config.forward_expansion,
            heads=config.heads,
            dropout=config.dropout,
            device=config.device,
            max_length=config.max_length
        ) for config in configs]
        self.fc_out = nn.Linear(configs[-1].max_length * configs[-1].trg_vocab_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, src, trg):
        out = self.tokenizer(src)["input_ids"]
        for transformer in self.transformers:
            out = transformer(out, trg)
        out = out.reshape(out.shape[0], -1)
        out = self.fc_out(out)
        out = self.tanh(out)
        return out