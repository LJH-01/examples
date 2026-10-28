# 图注意力网络

本仓库包含**图注意力网络（GAT）**的 PyTorch 实现，基于 Velickovic 等人的论文[《Graph Attention Network》](https://arxiv.org/abs/1710.10903v3)。

图注意力网络是一种强大的图神经网络模型，用于在图结构数据上学习表示，已在节点分类、链接预测和图分类等各种任务中展现出优异的性能。


## 概述
图注意力网络（GAT）是一种专门为处理图结构数据而设计的图神经网络架构。它利用多头注意力机制，以关注的方式捕获相邻节点的信息，从而为每个节点学习表示。这种注意力机制使模型能够关注相关节点，并在消息传递过程中自适应地为其贡献加权。

查看以下资源以了解更多关于 GAT 的信息：
- [主要作者 Petar Velickovic 的博客文章](https://petar-v.com/GAT/)
- [主要论文](https://doi.org/10.48550/arXiv.1710.10903)

本仓库提供了使用 PyTorch 对官方 GAT 模型的简洁短小的实现。代码注释完善、易于理解，对有兴趣研究图深度学习的研究人员和实践者来说是一份宝贵的资源。


## 主要特性

- **GAT 模型**：基于 Velickovic 等人论文《Graph Attention Network》的、带有多头注意力的图注意力网络模型实现。
- **图注意力层**：图卷积层的实现，使用自注意力机制聚合来自相邻节点的信息，以学习节点的重要性权重。
- **训练与评估**：在 *Cora* 基准数据集上，于图结构数据上训练 GAT 模型并评估其在节点分类任务上性能的代码。

---

# 环境要求
- Python 3.7 或更高版本
- PyTorch 2.0 或更高版本
- Requests 2.31 或更高版本
- NumPy 1.24 或更高版本



# 数据集
该实现包含对 Cora 数据集的支持，这是基于图的机器学习任务的标准基准数据集。Cora 数据集由科学出版物组成，其中节点代表论文，边代表引用关系。每篇论文都关联一个二值标签，指示其属于七个类别之一。该数据集已下载、预处理并可直接使用。

# 模型架构
Velickovic 等人在论文《Graph Attention Network》中提出的官方架构（本项目所采用的架构）由两个图注意力层组成，在其消息变换与聚合过程中融入了多头注意力机制。每个图注意力层对图中的每个节点应用共享的自注意力机制，使它们能够根据邻居的重要性学习不同的表示。

在激活函数方面，GAT 模型同时采用了 **指数线性单元（ELU）** 和 **带泄漏的修正线性单元（LeakyReLU）** 激活，为模型引入非线性。ELU 用作**隐藏层**的激活函数，而 LeakyReLU 应用于**注意力系数**，以确保负值也能获得非零梯度。

遵循官方实现，第一个 GAT 层由 **K = 8 个注意力头**组成，每个头计算 **F' = 8 个特征**（**共 64 个特征**），随后对该层输出应用指数线性单元（ELU）激活。第二个 GAT 层用于分类：**单个注意力头**计算 C 个特征（其中 C 为类别数），随后应用 softmax 激活以输出概率。（为便于配合 NLLLoss 进行计算，我们改用 log-softmax）

*请注意，由于这是一个教学示例，本实现使用图的邻接矩阵的完整稠密形式，而非稀疏形式。因此模型实现中的所有运算都以非稀疏形式完成。这不会影响模型在准确率方面的表现。不过，一个对稀疏形式友好的实现将有助于提升资源使用、存储和速度方面的效率。*
 

# 用法
在 Cora 数据集上训练和评估 GAT 模型可以通过如下方式运行 `main.py` 脚本完成：

1. 克隆 PyTorch 示例仓库：

```
git clone https://github.com/pytorch/examples.git
cd examples/gat
```

2. 安装所需的依赖：

```
pip install -r requirements.txt
```

3. 通过如下方式运行 `main.py` 脚本训练 GAT 模型：（使用默认参数的示例）

```bash
python main.py --epochs 300 --lr 0.005 --l2 5e-4 --dropout-p 0.6 --num-heads 8 --hidden-dim 64 --val-every 20
```

更详细地说，`main.py` 脚本接受以下参数：
```
usage: main.py [-h] [--epochs EPOCHS] [--lr LR] [--l2 L2] [--dropout-p DROPOUT_P] [--hidden-dim HIDDEN_DIM] [--num-heads NUM_HEADS] [--concat-heads] [--val-every VAL_EVERY]
               [--no-cuda] [--no-mps] [--dry-run] [--seed S]

PyTorch Graph Attention Network

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 300)
  --lr LR               learning rate (default: 0.005)
  --l2 L2               weight decay (default: 6e-4)
  --dropout-p DROPOUT_P
                        dropout probability (default: 0.6)
  --hidden-dim HIDDEN_DIM
                        dimension of the hidden representation (default: 64)
  --num-heads NUM_HEADS
                        number of the attention heads (default: 4)
  --concat-heads        wether to concatinate attention heads, or average over them (default: False)
  --val-every VAL_EVERY
                        epochs to wait for print training and validation evaluation (default: 20)
  --no-accel            disables accelerator
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 13)
```



# 结果
使用默认超参数在随机的训练/验证/测试数据划分上训练 **300 个 epoch** 后，GAT 模型在测试集上取得了约 **81.25%** 的分类准确率。该结果与原始论文中报告的性能相当。不过，由于训练/验证/测试划分的随机性，结果可能会有所波动。

# 参考文献

``` 
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
```
- arxiv 上的论文：[arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)
- 原始论文仓库：[https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)
