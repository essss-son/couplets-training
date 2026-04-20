import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .attention import MyMultiheadAttention
import copy


class Mytransformer_encoder_layer(nn.Module):
    def __init__(self, d_model, num_heads, forward_dim=2048, dropout=0.1):
        super(Mytransformer_encoder_layer, self).__init__()
        self.attn_output = MyMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, forward_dim)
        self.linear2 = nn.Linear(forward_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.attn_output(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = src + self.dropout1(x)
        x = self.norm1(x)

        output = self.activation(self.linear1(x))
        output = self.dropout2(output)
        output = self.linear2(output)
        output = x + output
        output = self.dropout3(output)
        output = self.norm2(output)

        return output


class Mytransformer_decoder_layer(nn.Module):
    def __init__(self, d_model, num_heads, forward_dim=2048, dropout=0.1):
        super(Mytransformer_decoder_layer, self).__init__()
        self.attn_output = MyMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, forward_dim)
        self.linear2 = nn.Linear(forward_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.attn_output(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # tgt2 = self.attn_output(tgt, memory, memory,
        #                           attn_mask=memory_mask, key_padding_mask=tgt_key_padding_mask)
        tgt2 = self.attn_output(tgt, memory, memory,
                                attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.dropout3(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout4(tgt2)

        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Mytransformer_encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Mytransformer_encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask, src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Mytransformer_decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Mytransformer_decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = output.norm()
        return output


class Mytransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers,
                 num_decoder_layers, feed_forward_dim=2048, dropout=0.1, norm=None):
        super(Mytransformer, self).__init__()
        encoder_layer = Mytransformer_encoder_layer(d_model, num_heads, feed_forward_dim, dropout)
        encoder_norm = norm
        self.encoder = Mytransformer_encoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = Mytransformer_decoder_layer(d_model, num_heads, feed_forward_dim, dropout)
        decoder_norm = norm
        self.decoder = Mytransformer_decoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()
        # self.d_model = d_model
        # self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask
