# NLP_Seq2Seq

基于 Seq2Seq 架构的英文-中文翻译模型，使用 GRU 编码器-解码器，支持 Teacher Forcing 训练。

## 项目结构

```
├── my_datasets/
│   ├── Tokenizer.py        # 分词器：英文分词 + 中文jieba分词 + 繁简转换 + 词表构建
│   └── Seq2SeqDataset.py   # 数据集：数据加载、预处理、collate_fn
├── model/
│   └── Seq2Seq.py          # 模型：Encoder (GRU) + Decoder (GRU + FC)
├── task/
│   └── translation.py      # 任务封装：训练循环、推理、损失计算
├── train.py                # 训练入口
├── infer.py                # 推理入口
├── data/                   # 训练数据 & 缓存
└── dict/                   # 词表 & 模型权重
```

## 环境要求

- Python 3.10
- PyTorch
- jieba
- opencc
- scikit-learn
- tqdm

## 快速开始

### 训练

```bash
python train.py
```

默认配置：`epochs=100`, `batch_size=128`, 当 loss < 0.1 时提前结束。

### 推理

```bash
python infer.py
```

修改 `infer.py` 中的输入句子即可翻译自定义英文。

## 模型架构

- **Encoder**: Embedding → Packed GRU → (output, hidden)
- **Decoder**: Embedding → GRU → Linear → 词表概率
- **训练**: 50% Teacher Forcing
- **推理**: Greedy decoding
