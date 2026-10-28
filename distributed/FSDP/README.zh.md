注意：FSDP1 已弃用。请参阅 [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)和[代码示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)。

## FSDP1 T5



要使用 FSDP1 运行用于文本摘要的 T5 示例：

## 获取 wikihow 数据集
```bash

sh download_dataset.sh

```

## 安装依赖：
~~~
pip install -r requirements.txt
~~~
## 确保你运行的是较新版本的 PyTorch：
参见 https://pytorch.org 以安装至少 1.12 版本，理想情况下安装当前的 nightly 构建。

使用 Torchrun 启动训练（将 nproc_per_node 调整为你的 GPU 数量）：

```
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

```
