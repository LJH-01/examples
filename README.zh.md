# PyTorch 示例

https://pytorch.org/examples/

`pytorch/examples` 是一个仓库，展示如何使用 [PyTorch](https://github.com/pytorch/pytorch) 的示例。其目标是提供精心挑选、短小、依赖极少（或无依赖）的 _高质量_ 示例，这些示例彼此之间差异显著，可以被借鉴到你现有的工作中。

- 教程请访问：https://github.com/pytorch/tutorials
- 对 pytorch.org 的修改请访问：https://github.com/pytorch/pytorch.github.io
- 通用模型中心请访问：https://pytorch.org/hub/ 或 https://huggingface.co/models
- 关于如何将 PyTorch 用于生产的实践指南请访问：https://github.com/facebookresearch/recipes
- 通用问答与支持请访问：https://discuss.pytorch.org/

## 可用模型

- [使用卷积网络的图像分类（MNIST）](./mnist/README.md)
- [使用 RNN 和 Transformer 的词级语言建模](./word_language_model/README.md)
- [使用流行网络训练 Imagenet 分类器](./imagenet/README.md)
- [生成对抗网络（DCGAN）](./dcgan/README.md)
- [变分自编码器](./vae/README.md)
- [使用高效亚像素卷积神经网络进行超分辨率重建](./super_resolution/README.md)
- [在 MNIST 上跨多个进程对共享 ConvNet 进行 Hogwild 训练](mnist_hogwild)
- [使用 actor-critic 训练 CartPole 保持平衡](./reinforcement_learning/README.md)
- [使用 GloVe 向量、LSTM 和 torchtext 的自然语言推理（SNLI）](snli)
- [时间序列预测——使用 LSTM 学习正弦波](./time_sequence_prediction/README.md)
- [在图像上实现神经风格迁移算法](./fast_neural_style/README.md)
- [在 OpenAI gym 上使用 Actor Critic 和 REINFORCE 算法进行强化学习](./reinforcement_learning/README.md)
- [使用 fx 进行 PyTorch 模块变换](./fx/README.md)
- 使用[分布式数据并行](./distributed/ddp/README.md)和 [RPC](./distributed/rpc) 的分布式 PyTorch 示例
- [若干演示 C++ 前端的示例](cpp)
- [使用 Forward-Forward 进行图像分类](./mnist_forward_forward/README.md)
- [使用 Transformer 进行语言翻译](./language_translation/README.md)

此外，还有一些托管在各自仓库中的优秀示例：

- [使用带注意力的序列到序列 RNN 进行神经机器翻译（OpenNMT）](https://github.com/OpenNMT/OpenNMT-py)

## 贡献

如果你想贡献自己的示例或修复 bug，请务必先查看 [CONTRIBUTING.md](CONTRIBUTING.md)。
