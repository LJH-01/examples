# 使用 RNN 和 Transformer 的词级语言建模

本示例在语言建模任务上训练一个多层 RNN（Elman、GRU 或 LSTM）或 Transformer。默认情况下，训练脚本使用随附的 Wikitext-2 数据集。
训练好的模型随后可以被生成脚本用来生成新的文本。

```bash
python main.py --accel --epochs 6           # Train a LSTM on Wikitext-2.
python main.py --accel --epochs 6 --tied    # Train a tied LSTM on Wikitext-2.
python main.py --accel --tied               # Train a tied LSTM on Wikitext-2for 40 epochs.
python main.py --accel --epochs 6 --model Transformer --lr 5
                                            # Train a Transformer model on Wikitext-2.
python main.py --accel --epochs 6 --model Transformer --use-optimizer --lr 0.001
                                            # Train a Transformer model with AdamW optimizer on Wikitext-2.

python generate.py --accel                  # Generate samples from the default model checkpoint.
```

> [!NOTE] 
> 本示例支持在加速设备（CUDA、MPS、XPU）上运行

该模型使用了 `nn.RNN` 模块（及其姊妹模块 `nn.GRU` 和 `nn.LSTM`）或 Transformer 模块（`nn.TransformerEncoder` 和 `nn.TransformerEncoderLayer`）；如果在安装了 cuDNN 的 CUDA 上运行，它们会自动使用 cuDNN 后端。

训练过程中，如果收到键盘中断（Ctrl-C），训练会停止，并会针对测试数据集对当前模型进行评估。

`main.py` 脚本接受以下参数：

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --accel               use accelerator
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the transformer model
  --dry-run             verify the code and the model
  --use-optimizer       specify whether to use an AdamW optimizer
```

借助这些参数，可以测试多种模型。
例如，以下参数会产生训练较慢但效果更好的模型：

```bash
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```
