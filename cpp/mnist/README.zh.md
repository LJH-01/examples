# 使用 PyTorch C++ 前端的 MNIST 示例

该文件夹包含一个示例：使用 PyTorch C++ 前端训练一个计算机视觉模型，以识别来自 MNIST 数据集图像中的数字。

全部训练代码都包含在 `mnist.cpp` 中。

要构建代码，请在终端中运行以下命令：

```shell
$ cd mnist
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应为解压后的 _LibTorch_ 发行版的路径，你可以从 [PyTorch
主页](https://pytorch.org/get-started/locally/)获取该发行版。

执行编译生成的二进制文件来训练模型：

```shell
$ ./mnist
Train Epoch: 1 [59584/60000] Loss: 0.4232
Test set: Average loss: 0.1989 | Accuracy: 0.940
Train Epoch: 2 [59584/60000] Loss: 0.1926
Test set: Average loss: 0.1338 | Accuracy: 0.959
Train Epoch: 3 [59584/60000] Loss: 0.1390
Test set: Average loss: 0.0997 | Accuracy: 0.969
Train Epoch: 4 [59584/60000] Loss: 0.1239
Test set: Average loss: 0.0875 | Accuracy: 0.972
...
```
