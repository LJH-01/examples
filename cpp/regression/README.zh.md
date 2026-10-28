# 线性回归示例

训练一个单层全连接层来拟合一个四次多项式。

要构建代码，请在终端中运行以下命令：

```shell
$ cd regression
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应为解压后的 _LibTorch_ 发行版的路径，你可以从 [PyTorch
主页](https://pytorch.org/get-started/locally/)获取该发行版。

执行编译生成的二进制文件来运行：

```shell
$ ./regression
Loss: 0.000301158 after 584 batches
==> Learned function:	y = 11.6441 x^4 -3.10164 x^3 2.19786 x^2 -3.83606 x^1 + 4.37066
==> Actual function:	y = 11.669 x^4 -3.16023 x^3 2.19182 x^2 -3.81505 x^1 + 4.38219
...
```
