# 🏮 对联生成 — 从零手写 Transformer

> 从零实现一个 Seq2Seq Transformer，完成英德翻译任务，为中文对联生成打下基础。

## 🏗 项目结构

```
couplets-training/
├── train.py                  # 训练入口
├── translate.py              # 推理 & 翻译脚本
├── config/
│   └── config.py             # 超参数（d_model, n_head, lr 等）
├── model/
│   ├── my_transformer.py     # Transformer Encoder + Decoder
│   ├── attention.py          # Multi-Head Self/Cross Attention
│   ├── translation_model.py  # Seq2Seq 封装
│   ├── Embedding.py          # Token Embedding + Positional Encoding
│   └── learning_rate.py      # Warmup + Decay 学习率调度
├── utils/
│   └── data_helpers.py       # 数据加载 & Tokenization
├── data/                     # 英德平行语料（Flickr）
│   ├── train.en / train.de
│   ├── val.en   / val.de
│   └── test_2016_flickr.en / test_2016_flickr.de
└── test/
    ├── test.py               # BLEU 评估
    ├── test_my_transformer.py
    ├── test_load_data.py
    └── new_test_tsf_ecd.py
```

## 🚀 快速开始

```bash
# 安装依赖
pip install torch spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# 小样本快速测试
python train.py --use_small

# 完整训练
python train.py

# 翻译推理
python translate.py
```

## 🧠 模型架构

| 组件 | 文件 | 说明 |
|------|------|------|
| Embedding | `model/Embedding.py` | Token Embedding + 正弦位置编码 |
| Multi-Head Attention | `model/attention.py` | 自注意力 + 交叉注意力 |
| Transformer | `model/my_transformer.py` | Encoder × 6 + Decoder × 6 |
| Seq2Seq | `model/translation_model.py` | Encoder-Decoder 封装 + 训练逻辑 |

## 📊 数据集

提供 **Flickr 8k 英德平行语料**（`train` / `val` / `test`），每行一句，制表符分隔。支持 `_small` 后缀的小样本用于快速调试。

## ⚙️ 关键配置

在 `config/config.py` 中调整：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `d_model` | 256 | 模型维度 |
| `n_head` | 8 | 注意力头数 |
| `n_layers` | 6 | Encoder/Decoder 层数 |
| `d_ff` | 1024 | FFN 隐藏层维度 |
| `dropout` | 0.1 | Dropout 比例 |
| `epochs` | 200 | 训练轮数 |

## 🗺 路线图

- [x] 英德翻译 Transformer
- [ ] 中文对联数据集构建
- [ ] 对联生成模型适配
- [ ] RLHF 优化对联质量
- [ ] 部署为在线服务

## 📄 License

MIT
