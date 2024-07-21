import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(self, query, keys, values, mask=None):
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, values)
        return output
    
    def split_heads(self, src):
        batch_size, seq_length, embed_size = src.size()
        return src.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, src):
        batch_size, _, seq_length, head_dim = src.size()
        return src.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)

    def forward(self, values, keys, query, mask=None):
        query = self.split_heads(self.queries(query))
        keys = self.split_heads(self.keys(keys))
        values = self.split_heads(self.values(values))

        out = self.scaled_dot_product_attention(query, keys, values, mask)
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
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_size, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask):
        attn_output = self.self_attn(src, src, src, mask)
        out = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(out)
        out = self.norm2(out + self.dropout(ff_output))
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.cross_attn = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_size, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(src, src, src, tgt_mask)
        out = self.norm1(src + self.dropout(attn_output))
        attn_output = self.cross_attn(out, enc_output, enc_output, src_mask)
        out = self.norm2(out + self.dropout(attn_output))
        ff_output = self.feed_forward(out)
        out = self.norm3(out + self.dropout(ff_output))
        return out

class Azsc_Transformer(nn.Module):
    def __init__(self, config):
        super(Azsc_Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(config.embed_size, config.max_length)
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.embed_size)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config.embed_size, config.num_heads, config.forward_expansion, config.dropout) for _ in range(config.num_layers)])
        if config.tgt_vocab_size > 0:
            self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.embed_size)
            self.decoder_layers = nn.ModuleList([DecoderLayer(config.embed_size, config.num_heads, config.forward_expansion, config.dropout) for _ in range(config.num_layers)])
            self.fc = nn.Linear(config.embed_size, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        if tgt is None:
            return src_mask, None
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt=None):
        device = src.device
        self.positional_encoding.to(device)
        self.encoder_embedding.to(device)
        self.encoder_layers.to(device)
        if tgt is not None:
            self.decoder_embedding.to(device)
            self.decoder_layers.to(device)
            self.fc.to(device)
        self.dropout.to(device)

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        # it will be None, but just to make it more verstile
        if tgt is None:
            return src_embedded
        
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output