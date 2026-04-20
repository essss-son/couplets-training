import sys

sys.path.append('../')
from model.my_transformer import Mytransformer
import torch


if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    src = torch.rand((src_len, batch_size, d_model))
    src_key_padding_mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, True],
    ])
    tgt = torch.rand((tgt_len, batch_size, d_model))

    tgt_key_padding_mask = torch.tensor([[False, False, False, True, True, True],
                                         [False, False, False, False, True, True]])

    model = Mytransformer(d_model, num_heads, num_encoder_layers,
                          num_decoder_layers, feed_forward_dim=64, norm=None)
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                   src_key_padding_mask=src_key_padding_mask,
                   memory_key_padding_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)
    print(output.size())
    print('ok')
