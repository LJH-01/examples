# 基于 PyTorch 的 SNLI 自然语言推理训练

## 📝 概述

本仓库包含用于训练自然语言推理（NLI）模型的 Python 脚本，具体为 `SNLIClassifier`，使用斯坦福自然语言推理（SNLI）语料库。训练好的模型用于预测文本蕴含关系，判断一个陈述是被另一个陈述蕴含、与之矛盾，还是两者皆非。

## ⚙️ 依赖

使用以下命令安装必要的 Python 库：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件包含：

```
torch
torchtext
spacy
```

## 💻 用法

使用以下命令启动训练过程：

```bash
python train.py --lower --word-vectors [PATH_TO_WORD_VECTORS] --vector-cache [PATH_TO_VECTOR_CACHE] --epochs [NUMBER_OF_EPOCHS] --batch-size [BATCH_SIZE] --save-path [PATH_TO_SAVE_MODEL] --gpu [GPU_NUMBER]
```

## 🏋️‍♀️ 训练

该脚本在指定数量的 epoch 内，基于小批量数据训练模型。它会在指定目录中将验证集上表现最好的模型保存为 `.pt` 文件。

## 📚 脚本

- `model.py`：定义 `SNLIClassifier` 模型及辅助类。
- `util.py`：包含用于创建目录和解析命令行参数的实用函数。

## 📣 注意

请确保 `model.py` 和 `util.py` 脚本位于你的工作目录中。
