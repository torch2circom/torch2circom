import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

import torch.optim as optim


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        # positional encoding
        max_len = 3
        emb_pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(10000.0, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim)
        div_term_odd = torch.pow(10000.0, torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim)

        emb_pe[:, 0::2] = torch.sin(position * div_term_even)
        emb_pe[:, 1::2] = torch.cos(position * div_term_odd)
        emb_pe = emb_pe.unsqueeze(0)
        # if CUDA == True:
        #     pe.type(torch.cuda.FloatTensor)
        self.register_buffer("emb_pe", emb_pe)  # pe: 5000, embed_dim

    def forward(self, x):
        return x + Variable(self.emb_pe[:, : x.size(1)], requires_grad=False)


class SingleheadAttn(nn.Module):
    def __init__(self, dim):
        super(SingleheadAttn, self).__init__()
        self.qw = nn.Linear(dim, dim)
        self.kw = nn.Linear(dim, dim)
        self.vw = nn.Linear(dim, dim)
        self.norm = L2NormalizationLayer(dim=-1)
        self.dim = dim

    def forward(self, q, k, v):
        query_in, key_in, value_in = q, k, v
        query = self.qw(q)  # B, N, d / d, d --> B, N, d
        key = self.kw(k)  # B, N, d / d, d --> B, N, d
        value = self.vw(v)  # B, N, d / d, d --> B, N, d
        key_transposed = torch.transpose(key, 1, 2)  # B, N, d --> B, d, N
        # Get attention weights
        attention_weights = torch.matmul(query, key_transposed)  # B, N, d / B, d, N --> B, N, N
        attention_weights = attention_weights / math.sqrt(self.dim)
        attention_weighted_value = torch.matmul(attention_weights, value)  # B, N, N / B, N, d --> B, N, d
        attention_weighted_value = attention_weighted_value

        attention_weighted_value = self.norm(attention_weighted_value)  # B, N, d --> B, N, d
        return attention_weighted_value


class AddandNorm(nn.Module):
    def __init__(self):
        super(AddandNorm, self).__init__()
        self.norm = L2NormalizationLayer(dim=-1)

    def forward(self, x, residual):
        x = self.norm(x + residual)  # B, N, d
        return x


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_blocks,
        num_heads,
        encoder_vocab_size,
        output_vocab_size,
        CUDA=False,
    ):
        super(TransformerTranslator, self).__init__()

        self.emb_encoder = nn.Embedding(encoder_vocab_size, embed_dim)  # vocab_size, embed_dim
        self.pe = PositionalEncoding(3, embed_dim)
        d_k = embed_dim
        d_v = embed_dim
        output_dim = embed_dim
        self.encoderattn = SingleheadAttn(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, output_dim)  # dim, dim
        self.RELU = nn.ReLU()
        self.linear_2 = nn.Linear(output_dim, embed_dim)  # dim, dim
        self.addandnorm = AddandNorm()

        self.d_k = d_k

        self.emb_decoder = nn.Embedding(output_vocab_size, embed_dim)  # out_vocab_size, embed_dim
        self.pe2 = PositionalEncoding(3, embed_dim)
        self.decoderselfattn = SingleheadAttn(embed_dim)
        self.linear_ds1 = nn.Linear(embed_dim, output_dim)
        self.RELU2 = nn.ReLU()
        self.linear_ds2 = nn.Linear(output_dim, embed_dim)
        self.addandnorm2 = AddandNorm()

        self.decodercrossattn = SingleheadAttn(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear_dc1 = nn.Linear(embed_dim, output_dim)
        self.RELU3 = nn.ReLU()
        self.linear_dc2 = nn.Linear(output_dim, embed_dim)
        self.addandnorm3 = AddandNorm()

        self.linear_vocab_logits = nn.Linear(embed_dim, output_vocab_size)

        self.encoded = False
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def encode(self, input_sequence):
        self.encoded = True

    def forward(self, input_sequence):
        input_sequence = input_sequence.to(torch.int)
        output_sequence = input_sequence
        x = self.emb_encoder(input_sequence)
        x = self.pe(x)
        residual_x = x
        attention_weighted_value = self.encoderattn(x, x, x)

        # feedforward
        x = attention_weighted_value
        x = self.linear_1(x)  # B, N, d / d, d --> B, N, d
        x = self.RELU(x)  # relu
        x = self.linear_2(x)  # B, N, d / d, d --> B, N, d
        x = self.dropout(x)
        x = self.addandnorm(x, residual_x)
        encoder_out = x

        x = self.emb_decoder(output_sequence)
        x = self.pe2(x)
        ds_residual_x = x[:, -1:, :]
        ds_attention_weighted_value = self.decoderselfattn(x[:, -1:, :], x, x)

        # decoder self attention feedforward
        x = ds_attention_weighted_value
        x = self.linear_ds1(x)
        x = self.RELU2(x)
        x = self.linear_ds2(x)
        x = self.dropout(x)
        x = self.addandnorm2(x, ds_residual_x)
        decoder_selfattn_out = x

        # decorder cross attention
        dc_residual_x = decoder_selfattn_out
        dc_attention_weighted_value = self.decodercrossattn(decoder_selfattn_out, encoder_out, encoder_out)
        dc_query_in, dc_key_in, dc_value_in = decoder_selfattn_out, encoder_out, encoder_out

        # decoder self attention feedforward
        x = dc_attention_weighted_value
        x = self.linear_dc1(x)
        x = self.RELU3(x)
        x = self.linear_dc2(x)
        x = self.dropout(x)
        x = self.addandnorm3(x, dc_residual_x)
        decoder_out = x

        out = self.linear_vocab_logits(decoder_out)  # B, N, d
        out = self.norm(out)
        return out
