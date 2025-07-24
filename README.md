这是我的大模型从基础到实战的2000行大作业其中之一，整体架构如下，共1515行代码

```plaintext
nano-gpt/                         # 项目根目录
    ├── data/                     # 数据相关（原始数据、预处理后的数据、词表等）
    │   ├── raw/                  # 原始数据集（如OpenWebText,wiki-zh，sci-fi）
    │   |   └── sci-fi/           # 收集的科幻小说数据集
    │   ├── processed/            # 预处理后的数据（token化后的pkl文件、训练/验证集拆分）
    │   |   └── sci-fi/           # 处理后的科幻小说数据
    │   |   |	|── meta.json     # 元数据
    │   |   |	|── train.npy     # 训练集
    │   |   |	└── val.npy       # 测试集
    │   └── vocab/            	  # 词表文件（对数据集进行分词后形成的词表）
    │   |   └── sci-fi.pkl        # 对科幻小说进行字符级分词后形成的词表
    ├── model/                    # 模型核心代码
    │   ├── __init__.py           # 模块初始化，方便导入
    │   └── gpt.py                # GPT模型主体（Transformer结构、注意力机制等） -225行代码
    │
    ├── train/                    # 训练相关代码
    │   ├── __init__.py
    │   └──  configs/             # 训练配置文件（不同模型规模、数据集的参数）
    │       └── default.py        # 训练的默认配置 -65行代码
    │
    ├── utils/                    # 通用工具函数
    │   ├── __init__.py
    │   └── tokenizer_utils.py    # 分词器工具（编码、解码、可通过字符级分词或jieba构建词表）-217行
    │
    ├── scripts/                  # 可执行脚本（方便快速运行训练、采样等）
    │   ├── __init__.py
    │   ├── prepare_data.py       # 数据预处理脚本（从原始文本生成token化数据）-123行
    │   ├── bench.py              # 评估模型在特定硬件环境下的训练性能,帮助优化配置参数 -148行
    │   ├── train.py              # 训练入口脚本（解析配置、启动训练器），可从头训和续训 -436行
    │   ├── sample.py             # 采样/推理脚本（从训练好的模型生成文本,生成md文件）-122行
    │   └── view_model.py         # 可以查看已经训练好的模型的一些关键参数 -114行
    │
    ├── experiments/              # 实验结果（模型checkpoint）
    │   └── nanoGPT-2/            # 某一次实验
    │       └── checkpoints.pt    # 保存的模型权重
    ├── inference/                # 样本生成的结果
    │   |── config/               # 用于生成样本的参数
    │       └── default.py        # 生成的默认参数 -65行
    |   |──nanoGPT-220250716-0947.md  # 生成的样本
    |   └──nanoGPT-220250716-2256.md
    │
    ├── requirements.txt          # 项目依赖（PyTorch、numpy、tokenizers等）
    └── README.md                 # 项目说明（环境配置、运行步骤、文件夹结构解释）
```

