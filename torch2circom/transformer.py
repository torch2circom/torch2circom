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
    def __init__(self, max_len):
        super(PositionalEncoding, self).__init__()
        # positional encoding
        max_len = 5000
        pe = torch.zeros(max_len, embed_dim).to('cuda')
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(
            10000.0, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        div_term_odd = torch.pow(
            10000.0, torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)
        if CUDA == True:
            pe.type(torch.cuda.FloatTensor)
        self.register_buffer("pe", pe) # pe: 5000, embed_dim

    def forward(self, x):
        return x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
    
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
        query = self.qw(q) # B, N, d / d, d --> B, N, d
        key = self.kw(k) # B, N, d / d, d --> B, N, d
        value = self.vw(v) # B, N, d / d, d --> B, N, d
        key_transposed = torch.transpose(key, 1, 2) # B, N, d --> B, d, N
        # Get attention weights
        attention_weights = torch.matmul(query, key_transposed)  # B, N, d / B, d, N --> B, N, N
        attention_weights = attention_weights / math.sqrt(self.dim)
        attention_weighted_value = torch.matmul(
            attention_weights, value
        )  # B, N, N / B, N, d --> B, N, d
        attention_weighted_value = attention_weighted_value
        
        
        attention_weighted_value = self.norm(attention_weighted_value) # B, N, d --> B, N, d
        return attention_weighted_value
    
class AddandNorm(nn.Module):
    def __init__(self):
        super(AddandNorm, self).__init__()
        self.norm = L2NormalizationLayer(dim=-1)
    def forward(self, x, residual):
        x = self.norm(x + residual) # B, N, d 
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

        # self.encoder_embedding = Embeddings(encoder_vocab_size, embed_dim, CUDA=CUDA)
        # self.output_embedding = Embeddings(output_vocab_size, embed_dim, CUDA=CUDA)
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_dim) # vocab_size, embed_dim
        self.pe = PositionalEncoding(5000)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_dim) # out_vocab_size, embed_dim
        
        # transformer block
            # MultiHeadAttention
        # self.attention_blocks = SelfAttention(embed_dim, d_k, d_v, mask)
                # Self attention
        d_k = embed_dim
        d_v = embed_dim
        output_dim=embed_dim
        self.encoderattn = SingleheadAttn(embed_dim)
        self.addandnorm = AddandNorm()
        self.d_k = d_k
        self.dropout = nn.Dropout(0.1)
        # self.normalization = L2NormalizationLayer(dim=-1)
        self.norm = L2NormalizationLayer(dim=-1)
        
        self.l1 = nn.Linear(embed_dim, output_dim) # dim, dim
        self.RELU = nn.ReLU()
        self.l2 = nn.Linear(output_dim, embed_dim) # dim, dim
        self.dropout = nn.Dropout(0.1)

        self.decoderselfattn = SingleheadAttn(embed_dim)
        # self.ds_query_embed = nn.Linear(embed_dim, d_k)
        # self.ds_key_embed = nn.Linear(embed_dim, d_k)
        # self.ds_value_embed = nn.Linear(embed_dim, d_v)

        self.ds_l1 = nn.Linear(embed_dim, output_dim)
        self.ds_l2 = nn.Linear(output_dim, embed_dim)
        
        self.decodercrossattn = SingleheadAttn(embed_dim)
        # self.dc_query_embed = nn.Linear(embed_dim, d_k)
        # self.dc_key_embed = nn.Linear(embed_dim, d_k)
        # self.dc_value_embed = nn.Linear(embed_dim, d_v)

        self.dc_l1 = nn.Linear(embed_dim, output_dim)
        self.dc_l2 = nn.Linear(output_dim, embed_dim)
        
        self.vocab_logits = nn.Linear(embed_dim, output_vocab_size)
        
        self.encoded = False
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def encode(self, input_sequence):
        self.encoded = True

    def forward(self, input_sequence):
        input_sequence = input_sequence.to(torch.int)
        
        output_sequence = input_sequence
        print(input_sequence.size())

        x = self.encoder_embedding(input_sequence).to(self.device)
        
        print(x.size())

        x = self.pe(x)
        residual_x = x
        attention_weighted_value = self.encoderattn(x, x, x)
        
        # feedforward
        x = attention_weighted_value
        x = self.l1(x) #B, N, d / d, d --> B, N, d
        x = self.RELU(x) # relu
        x = self.l2(x) #B, N, d / d, d --> B, N, d
        x = self.dropout(x)
        x = self.addandnorm(x, residual_x)
        encoder_out = x

        x = self.decoder_embedding(output_sequence).to(self.device)
        x = self.pe(x)
        ds_residual_x = x[:, -1:, :]
   
        # x: B, N, d
        # ds_query_in: B, 1, d
        ds_attention_weighted_value = self.decoderselfattn(x[:, -1:, :], x, x)
        # decoder self attention feedforward
        x = ds_attention_weighted_value
        x = self.ds_l1(x)
        x = self.RELU(x)
        x = self.ds_l2(x)
        x = self.dropout(x)
        x = self.addandnorm(x, ds_residual_x)
        decoder_selfattn_out = x
        
        
        # decorder cross attention
        dc_residual_x = decoder_selfattn_out
        dc_attention_weighted_value = self.decodercrossattn(decoder_selfattn_out, encoder_out, encoder_out)
        dc_query_in, dc_key_in, dc_value_in = decoder_selfattn_out, encoder_out, encoder_out

        
        # decoder self attention feedforward
        x = dc_attention_weighted_value
        x = self.dc_l1(x)
        x = self.RELU(x)
        x = self.dc_l2(x)
        x = self.dropout(x)
        x = self.addandnorm(x, dc_residual_x)
        decoder_out = x
        
        out = self.vocab_logits(decoder_out) # B, N, d
        out = self.norm(out)
        # import ipdb
        # ipdb.set_trace()
        # out = torch.max(out, dim=-1) # B, N
        # import ipdb
        # ipdb.set_trace()
        return out