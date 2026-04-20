from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn
import torch

is_print_shape = False

class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention_forward(query, key, value, self.num_heads,
                                             self.dropout,
                                             out_proj=self.out_proj,
                                             training=self.training,
                                             key_padding_mask=key_padding_mask,
                                             q_proj=self.q_proj,
                                             k_proj=self.k_proj,
                                             v_proj=self.v_proj,
                                             attn_mask=attn_mask,
                                             )
    def multi_head_attention_forward(self,
                                    query,
                                    key,
                                    value,
                                    num_heads,
                                    dropout_p,
                                    out_proj,
                                    training=True,
                                    key_padding_mask=None,
                                    q_proj=None,
                                    k_proj=None,
                                    v_proj=None,
                                    attn_mask=None,
                                    ):
        q = q_proj(query)  # [tgt_len,batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]
        k = k_proj(key)  # [src_len, batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [src_len,batch_size,kdim * num_heads]
        v = v_proj(value)  # [src_len, batch_size, embed_dim] x [embed_dim,vdim * num_heads]  = [src_len, batch_size, vdim * num_heads]

        tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size,embed_dim]

        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim == 2:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise ValueError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz*num_heads, query.size(0), key.size(0)]:
                    raise ValueError('The size of the 3D attn_mask is not correct.')

        q = q.contiguous().view(tgt_len, bsz*num_heads, head_dim).transpose(0, 1) #[batch_size * num_heads,tgt_len,kdim]
        k = k.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0, 1) #[batch_size * num_heads,src_len,kdim]
        v = v.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0, 1) #[batch_size * num_heads,src_len,vdim]

        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
                                                              # = [batch_size * num_heads,tgt_len,src_len]
        # print('=============attention里===================')
        # print(attn_output_weights.device, attn_mask.device)
        # print('=============attention里===================')
        if attn_mask is not None:
            attn_output_weights += attn_mask    # [batch_size*num_heads,tgt_len,src_len]
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)  # 变成[batch_size, num_heads, tgt_len, src_len]的形状

            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -float('inf'))
            # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]

            attn_output_weights = attn_output_weights.view(bsz*num_heads, tgt_len, src_len)
            # [batch_size * num_heads, tgt_len, src_len]
        attn_output_weights = F.softmax(attn_output_weights, dim=-1) # [batch_size * num_heads, tgt_len, src_len]
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v) # Z=[batch_size*num_heads,tgt_len, src_len]@[batch_size * num_heads,src_len,vdim]
                                                        # = [batch_size * num_heads,tgt_len,vdim]

        attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len, bsz, embed_dim)

        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        Z = out_proj(attn_output) # 这里就是多个z  线性组合成Z  [tgt_len,batch_size,embed_dim]
        if is_print_shape:
            print(f"\t 多头注意力中，多头计算结束后的形状堆叠为([tgt_len, batch_size, num_heads*head_dim]):{attn_output.size()}")
            print(f"\t 多头计算结束后，再进行线性变换时的权重W_o的形状为([num_heads*head_dim, num_heads*head_dim]):{out_proj.weight.size()}")
            print(f"\t 多头线性变化后的形状为([tgt_len, batch_size, embed_dim]):{Z.size()}")
        return Z,  attn_output_weights.sum(dim=1) / num_heads

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
    # my_multihead_attention_encoder = MyMultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    # r = my_multihead_attention_encoder(src, src, src, key_padding_mask=src_key_padding_mask)
    # print(r[0].shape)

    my_multihead_attention_decoder = MyMultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    r = my_multihead_attention_decoder(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)

    multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    r = multihead_attention(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)
