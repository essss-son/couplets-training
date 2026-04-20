import sys
sys.path.append('../')
from model import MyMultiheadAttention  #之前的测试

from model import MyTransformerEncoderLayer, MyTransformerDecoderLayer
from model import MyTransformerEncoder, MyTransformerDecoder
import torch
import torch.nn as nn

if __name__ == '__main__':

    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_heads = 8
    src = torch.rand((src_len, batch_size, d_model))
    src_key_padding_mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, True],
    ])
    tgt = torch.rand((tgt_len, batch_size, d_model))
    my_multihead_attention = MyMultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    r = my_multihead_attention(src, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)

    my_multihead_attention = MyMultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    r = my_multihead_attention(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)

    my_multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    r = my_multihead_attention(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)



    # src_len = 5
    # batch_size = 2
    # d_model = 32
    # tgt_len = 6
    # num_head = 8
    # num_layers = 2
    # src = torch.rand((src_len, batch_size, d_model))
    # src_key_padding_mask = torch.tensor([
    #     [False, False, False, True, True],
    #     [False, False, False, False, True],
    # ])
    # tgt = torch.rand((tgt_len, batch_size, d_model))
    # tgt_key_padding_mask = torch.tensor([
    #     [False, False, False, True, True, True],
    #     [False, False, False, False, True, True],
    # ])
    #
    # tgt = torch.rand((tgt_len, batch_size, d_model))
    # encoder_layer = MyTransformerEncoderLayer(d_model, num_head, dim_feedforward=128) # 32 8 128
    # encoder = MyTransformerEncoder(encoder_layer, num_layers)
    # memory = encoder(src, src_key_padding_mask=src_key_padding_mask)
    #
    # decoder_layer = MyTransformerDecoderLayer(d_model, num_head, dim_feedforward=128)
    # decoder = MyTransformerDecoder(decoder_layer, num_layers)
    # result = decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask,
    #                  memory_key_padding_mask=src_key_padding_mask)
    #
    # print(result.shape)
