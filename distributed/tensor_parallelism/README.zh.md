# 用于分布式训练的 PyTorch 原生张量并行

本示例通过使用 PyTorch 原生张量并行 API 来演示 SPMD Megatron-LM 风格的张量并行，包括：

1. 在虚拟 MLP 模型上的简单模块级张量并行。
2. 在虚拟 MLP 模型上的、带序列并行输入/输出的简单模块级张量并行。
3. 在示例 Llama2 模型上完全分片数据并行 + 张量并行（含序列并行）的端到端演示。

关于 PyTorch 原生张量并行 API 的更多细节，请参阅 PyTorch 文档：
https://pytorch.org/docs/stable/distributed.tensor.parallel.html

## 安装

```bash
pip install -r requirements.txt
```

## 运行示例

你可以使用 `torchrun` 启动分布式训练来运行这些示例：

```bash
# Simple Tensor Parallel example
torchrun --nnodes=1 --nproc_per_node=4 tensor_parallel_example.py

# Tensor Parallel with Sequence Parallel
torchrun --nnodes=1 --nproc_per_node=4 sequence_parallel_example.py

# FSDP + Tensor Parallel with Llama2 model
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

更多细节请查看 `run_examples.sh` 脚本。
