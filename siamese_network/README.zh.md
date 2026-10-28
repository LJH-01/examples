# 孪生网络示例
用于图像相似度估计的孪生网络。
该网络由两个完全相同的网络组成，每个输入对应其中一个。
每个网络的输出会被拼接起来并传入一个线性层。
线性层的输出再经过一个 sigmoid 函数。
[FaceNet](https://arxiv.org/pdf/1503.03832.pdf) 是孪生网络的一种变体。
本实现与 FaceNet 有所不同，因为我们使用
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) 中的 `ResNet-18` 模型作为特征提取器。
此外，由于 MNIST 数据集较为简单，我们没有使用 `TripletLoss`，`BCELoss` 就足够了。

### 用法

安装所需的依赖：
```bash
pip install -r requirements.txt
```

要运行该示例，请执行：
```bahs
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

如果检测到硬件加速设备，该示例将在加速器上执行；否则将在 CPU 上运行。

要强制在 CPU 上执行，请使用 `--no-accel` 命令行参数：

```bash
python main.py --no-accel
```

可选地，你可以添加以下参数来自定义执行方式。

```bash
--batch-size            input batch size for training (default: 64)
--test-batch-size       input batch size for testing (default: 1000)
--epochs                number of epochs to train (default: 14)
--lr                    learning rate (default: 1.0)
--gamma                 learning rate step gamma (default: 0.7)
--no-accel              disables accelerator
--dry-run               quickly check a single pass
--seed                  random seed (default: 1)
--log-interval          how many batches to wait before logging training status
--save-model            Saving the current Model
```
