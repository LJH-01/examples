# 使用 PyTorch C++ 前端的 DCGAN 示例

该文件夹包含一个使用 PyTorch C++ 前端训练 DCGAN 以生成 MNIST 数字的示例。

全部训练代码都包含在 `dcgan.cpp` 中。

你可以[在此处](https://github.com/pytorch/examples/blob/main/.github/workflows/main_cpp.yml#L34)找到安装 argparse 的命令。

要构建代码，请在终端中运行以下命令：

```shell
$ cd dcgan
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应为解压后的 _LibTorch_ 发行版的路径，你可以从 [PyTorch
主页](https://pytorch.org/get-started/locally/)获取该发行版。

执行编译生成的二进制文件来训练模型：

```shell
$ ./dcgan
[ 1/30][200/938] D_loss: 0.4953 | G_loss: 4.0195
-> checkpoint 1
[ 1/30][400/938] D_loss: 0.3610 | G_loss: 4.8148
-> checkpoint 2
[ 1/30][600/938] D_loss: 0.4072 | G_loss: 4.36760
-> checkpoint 3
[ 1/30][800/938] D_loss: 0.4444 | G_loss: 4.0250
-> checkpoint 4
[ 2/30][200/938] D_loss: 0.3761 | G_loss: 3.8790
-> checkpoint 5
[ 2/30][400/938] D_loss: 0.3977 | G_loss: 3.3315
-> checkpoint 6
[ 2/30][600/938] D_loss: 0.3815 | G_loss: 3.5696
-> checkpoint 7
[ 2/30][800/938] D_loss: 0.4039 | G_loss: 3.2759
-> checkpoint 8
[ 3/30][200/938] D_loss: 0.4236 | G_loss: 4.5132
-> checkpoint 9
[ 3/30][400/938] D_loss: 0.3645 | G_loss: 3.9759
-> checkpoint 10
...
```

我们也可以指定 `--epochs` 来改变训练的 epoch 数，如下所示：

```shell
$ ./dcgan --epochs 10
```
如果不指定 `--epochs` 标志，默认训练的 epoch 数为 30。


训练脚本会定期生成图像样本。使用位于该文件夹中的
`display_samples.py` 脚本生成绘图图像。
例如：

```shell
$ python display_samples.py -i dcgan-sample-10.pt
Saved out.png
```
