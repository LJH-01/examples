# 使用 Libtorch 和 OpenCV 在 Dogs vs Cats 数据集上进行迁移学习

使用 PyTorch C++ API 在 Dogs vs Cats 数据集上进行迁移学习。

## 用法

**训练**：

1. 移除 `ResNet18` 预训练模型的最后一层，并转换为 `torch.jit` 模块：`python3 convert.py`。
2. 创建构建目录：`mkdir build && cd build`
3. `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`
4. `make`
5. 运行训练代码：`./example <path_to_scripting_model>`

**预测**：

1. `cd build`
2. `./classify <path_image> <path_to_resnet18_model_without_fc_layer> <model_linear_trained>` : `./classify <path_image> ../resnet18_without_last_layer.pt model_linear.pt`

关于使用 Libtorch 应用迁移学习的详细博客：https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/。
