import torch.nn as nn
import torch
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)  # 权重矩阵 vocab_size * emb_size 行列
        # self.emb_size = emb_size

    def forward(self, tokens):  # tokens的形状  [seq_len, batch_size]
        return self.embedding(tokens)  # [seq_len, batch_size, emb_size]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # 经过这一句之后pe的具体内容是[max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)

if __name__ == '__main__':
    src = torch.tensor([[1, 2, 3],
                        [1, 2, 4]])
    token = TokenEmbedding(10, 20)
    x = token(src.transpose(0, 1))
    # x:  [3, 2, 20]

    position = PositionalEmbedding(20)
    y = position(x)

    print(x)
    print('================PositionEmbedding后=====================')
    print(y)