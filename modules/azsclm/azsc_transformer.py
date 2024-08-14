import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.device = device

        assert (
            embed_size % num_heads == 0
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(embed_size, embed_size, bias=False).to(self.device)
        self.keys = nn.Linear(embed_size, embed_size, bias=False).to(self.device)
        self.values = nn.Linear(embed_size, embed_size, bias=False).to(self.device)
        self.fc_out = nn.Linear(embed_size, embed_size).to(self.device)

    def scaled_dot_product_attention(self, query, keys, values, mask=None):
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, values)
        return output
    
    def split_heads(self, src):
        batch_size, seq_length, embed_size = src.size()
        out = src.view(batch_size, seq_length, self.num_heads, self.head_dim)
        out = out.transpose(1, 2)
        return out
    
    def combine_heads(self, src):
        batch_size, num_heads, seq_length, head_dim = src.size()
        out = src.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)
        return out

    def forward(self, values, keys, query, mask=None):
        query = self.split_heads(self.queries(query))
        keys = self.split_heads(self.keys(keys))
        values = self.split_heads(self.values(values))

        #out = self.scaled_dot_product_attention(query, keys, values, mask)
        out = self.fc_out(self.combine_heads(out))

        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion)
        self.fc2 = nn.Linear(forward_expansion, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, src):
        return src + self.pe[:, :src.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.device = config.device
        self.self_attn = MultiHeadAttention(config.embed_size, config.num_heads, config.device).to(self.device)
        self.feed_forward = PositionWiseFeedForward(config.embed_size, config.forward_expansion).to(self.device)
        self.norm1 = nn.LayerNorm(config.embed_size).to(self.device)
        self.norm2 = nn.LayerNorm(config.embed_size).to(self.device)
        self.dropout = nn.Dropout(config.dropout).to(self.device)
        
    def forward(self, src, mask=None):
        attn_output = self.self_attn(src, src, src, mask)
        out = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(out)
        out = self.norm2(out + self.dropout(ff_output))
        return out
    
class Azsc_Transformers(nn.Module):
    def __init__(self, config):
        super(Azsc_Transformers, self).__init__()
        self.device = config.device
        self.positional_encoding = PositionalEncoding(config.embed_size, config.max_length).to(self.device)
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.embed_size).to(self.device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)]).to(self.device)
        self.dropout = nn.Dropout(config.dropout).to(self.device)
        self.fc = nn.Linear(config.embed_size, config.tgt_size).to(self.device)

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):

        src_mask = self.generate_mask(src)
        out = self.encoder_embedding(src)
        out = self.positional_encoding(out)
        out = self.dropout(out)
        for enc_layer in self.encoder_layers:
            out = enc_layer(out, src_mask)
        out = self.fc(out)
        
        return out
        