# 语言翻译

本示例展示了如何使用 transformer 进行语言翻译。具体而言，该实现大致基于[《Attention is All You Need》论文](https://arxiv.org/abs/1706.03762)。

## 环境要求

我们需要针对相应语言的分词器。Torchtext 确实包含英语的分词器，但遗憾的是，我们需要的语言不止一种。我们可以通过 ```spacy``` 获取这些分词器

```bash
python3 -m spacy download <language>
python3 -m spacy download en
python3 -m spacy download de
```

Spacy 支持多种语言。关于受支持语言的完整清单，请查看[此处](https://spacy.io/usage/models)。本示例默认从德语翻译为英语。

还需要 Torchtext：
```bash
pip install torchtext
```

只需运行以下命令即可开始：
```bash
pip install -r requirements.txt
python3 -m spacy download <language-you-want>
```

## 用法

本示例包含许多标志，你可以设置它们来改变模块的行为/训练方式。你可以通过运行以下命令查看所有标志：

```bash
python3 main.py -h
```

但一般来说，所有设置都有"合理"的默认值；不过，默认的翻译方向是从德语到英语。要*训练*模型，你只需运行以下命令，同时也给出了如何使用任意语言的示例：

```bash
python3 main.py
python3 main.py --src en --tgt fr # For english to french translation
```

对于模型推理，你可以使用以下命令：

```bash
python3 main.py --inference --model_path <path-to-model>
```

经过一段加载时间后，这会打开一个交互式界面，你可以在其中输入任何你想要翻译的句子。
