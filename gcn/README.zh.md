# 图卷积网络

本仓库包含图卷积网络（GCN）的实现，基于 Thomas N. Kipf 和 Max Welling 的论文《Semi-Supervised Classification with Graph Convolutional Networks》。

## 概述
本项目实现了论文中提出的 GCN 模型，用于图结构数据上的半监督节点分类。GCN 利用图卷积聚合来自相邻节点的信息，并学习用于下游任务的节点表示。该实现为基于图的机器学习任务提供了一个灵活且高效的 GCN 模型。

## 环境要求
```bash
pip install -r requirements.txt
```

# 用法
```bash
python main.py --epochs 200 --lr 0.01 --l2 5e-4 --dropout-p 0.5 --hidden-dim 16 --val-every 20 --include-bias
```

# 数据集
该实现包含对 Cora 数据集的支持，这是基于图的机器学习任务的标准基准数据集。Cora 数据集由科学出版物组成，其中节点代表论文，边代表引用关系。每篇论文都关联一个二值标签，指示其属于七个类别之一。该数据集已下载、预处理并可直接使用。

## 模型架构
GCN 模型架构遵循论文中提供的细节。它由多个带 ReLU 激活的图卷积层组成，最后接一个用于分类的 softmax 层。该实现支持可定制的超参数，例如隐藏单元数、层数和 dropout 率。

# 结果
经过 200 个 epoch 的训练后，该模型在 Cora 数据集的测试集上达到了 82.5% 的分类准确率。该结果与原始论文中报告的性能相当。不过，由于训练/验证/测试划分的随机性，结果可能会有所波动。

参考文献
Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks." Link to the paper

原始论文仓库：[https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)
