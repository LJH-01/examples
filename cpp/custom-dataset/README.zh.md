# 使用 PyTorch C++ 前端的自定义数据集示例

该文件夹包含一个示例：使用 OpenCV 加载自定义图像数据集，并使用 PyTorch C++ 前端训练一个为图像打标签的模型。

这里使用的数据集是 [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集。

全部训练代码都包含在 custom-data.cpp 中。

你可以[在此处](../tools/InstallingOpenCV.md)找到安装 OpenCV 的说明。

要构建代码，请在终端中运行以下命令：

```shell
$ cd custom-dataset
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 /path/to/libtorch 应为解压后的 LibTorch 发行版的路径，你可以从 [PyTorch 主页](https://pytorch.org/get-started/locally/)获取该发行版。

如果在运行 `make` 命令时看到类似 `undefined reference to cv::imread(std::string const&, int)` 的错误，你应按照[此处](https://github.com/pytorch/pytorch#from-source)的说明从源码构建 LibTorch，然后将 `CMAKE_PREFIX_PATH` 设置为该 PyTorch 源码目录。另一种解决方案是使用 `libtorch-cxx11-abi-shared-with-deps` 而不是 `libtorch-shared-with-deps`，因为后者与 openCV 不兼容（参见[此处](https://discuss.pytorch.org/t/library-conflict-between-libtorch-and-opencv/64489)的报告）。

构建目录应如下所示：

```
.
├── custom-dataset
├── dataset
│   ├── accordion
│   │   ├── image_0001.jpg
│   │   ├── ...
│   ├── airplanes
│   │   ├── ...
│   ├── ...
├── info.txt
└── Makefile
└── ...
```

`info.txt` 文件会在构建过程中从源码目录复制过来。

执行编译生成的二进制文件来训练模型：

```shell
./custom-dataset
Running on: CUDA
Train Epoch: 1 16/7281	Loss: 0.314655	Acc: 0
Train Epoch: 1 176/7281	Loss: 0.532111	Acc: 0.0681818
Train Epoch: 1 336/7281	Loss: 0.538482	Acc: 0.0714286
Train Epoch: 1 496/7281	Loss: 0.535302	Acc: 0.0705645
Train Epoch: 1 656/7281	Loss: 0.536113	Acc: 0.0716463
Train Epoch: 1 816/7281	Loss: 0.537626	Acc: 0.0784314
Train Epoch: 1 976/7281	Loss: 0.537055	Acc: 0.079918
...

```
