import torch.nn as nn
import torch
from model.Embedding import TokenEmbedding, PositionalEmbedding
from model.my_transformer import Mytransformer


class Translation_model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(Translation_model, self).__init__()
        self.src_token_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, d_model)  # 仔细理解一下这里的词表
        self.positional_embedding = PositionalEmbedding(d_model, dropout)
        self.transformer = Mytransformer(d_model, num_heads, num_encoder_layers=num_encoder_layers,
                                         num_decoder_layers=num_decoder_layers, feed_forward_dim=dim_feedforward,
                                         dropout=dropout)
        self.classifier = nn.Linear(d_model, tgt_vocab_size)  # 注意！！！这里是tgt_vocab_size

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask):
        src = self.src_token_embedding(src)
        src = self.positional_embedding(src)
        tgt = self.tgt_token_embedding(tgt)
        tgt = self.positional_embedding(tgt)
        output = self.transformer(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        # output: [tgt_len, batch_size, embed_dim]
        logits = self.classifier(output)
        return logits
        # logits: [tgt_len, batch_size, tgt_vocab_size]

    def encoder(self, src):
        src_embed = self.src_token_embedding(src)
        src_embed = self.positional_embedding(src_embed)
        memory = self.transformer.encoder(src_embed)
        return memory

    def decoder(self, tgt, memory):
        tgt_embed = self.tgt_token_embedding(tgt)
        tgt_embed = self.positional_embedding(tgt_embed)
        outs = self.transformer.decoder(tgt_embed, memory)
        return outs


if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_heads = 8
    src = torch.tensor([[1, 2, 3, 4, 5],
                        [1, 2, 5, 7, 9]]).transpose(0, 1)
    src_key_padding_mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, True],
    ])
    tgt = torch.tensor([[1, 2, 3, 4, 5, 6],
                        [1, 2, 6, 2, 8, 9]]).transpose(0, 1)
    tgt_key_padding_mask = torch.tensor([[False, False, False, True, True, True],
                                         [False, False, False, False, True, True]])
    trans_model = Translation_model(src_vocab_size=20, tgt_vocab_size=20, d_model=d_model, num_heads=num_heads,
                                    num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=64, dropout=0.1)
    tgt_mask = trans_model.transformer.generate_square_subsequent_mask(tgt_len)
    logits = trans_model(src, tgt, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                         src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(logits.size())
    print('ok')
