分布式流水线并行示例

本示例展示如何将 ResNet50 模型分布到两个 RPC worker 上，
然后使用 RPC 实现分布式流水线并行。在流水线
并行中，每个输入批次被划分为多个微批次（micro-batch），这些
微批次以流水线的方式送入模型，以提高
设备的摊薄利用率。请注意，本示例只并行化
前向传播，可以视为
[单机流水线并行](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)
示例的分布式版本。

```
pip install -r requirements.txt
python main.py
```
