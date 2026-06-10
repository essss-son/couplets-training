# 🏮 对联生成 — 从零手写 Transformer

> 从零实现 Seq2Seq Transformer，完成英德翻译任务。

## 🏗 项目结构

```
couplets-training/
├── train.py                  # 训练入口
├── translate.py              # 推理脚本
├── config/config.py           # 超参数
├── model/
│   ├── my_transformer.py     # Encoder + Decoder
│   ├── attention.py          # Multi-Head Attention
│   ├── translation_model.py  # Seq2Seq
│   ├── Embedding.py          # 词嵌入 + 位置编码
│   └── learning_rate.py      # 学习率调度
├── utils/data_helpers.py     # 数据加载
├── data/                     # Flickr 8k 英德平行语料
└── test/
```

## 🚀 快速开始

```bash
pip install torch spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python train.py --use_small   # 小样本快速测试
python train.py               # 完整训练
python translate.py           # 推理
```

## 🧠 模型架构

| 组件 | 说明 |
|------|------|
| Embedding | Token Embedding + 正弦位置编码 |
| Attention | Multi-Head Self/Cross Attention |
| Transformer | Encoder x6 + Decoder x6 |
| Seq2Seq | Encoder-Decoder 封装 |

## 📄 License
MIT
