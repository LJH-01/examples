# 使用 PyTorch C++ 前端（Libtorch）在 MNIST 上进行分布式训练

该文件夹包含一个在 MNIST 数据集上对卷积神经网络进行数据并行训练的示例。并行化使用了消息传递接口（MPI）。

全部代码都包含在 dist-mnist.cpp 中

你可以[在此处] (https://www.open-mpi.org/faq/?category=building)找到安装 MPI 的说明。该代码已在 Open MPI 上测试，但应该也能在其他 MPI 发行版（如 MPICH、MVAPICH 等）上运行。

要构建代码，请在终端中运行以下命令：

```shell
$ cd distributed
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 /path/to/libtorch 应为解压后的 LibTorch 发行版的路径。请注意，来自 [PyTorch 主页] ((https://pytorch.org/get-started/locally/) 的 LibTorch 不包含 MPI 头文件，无法用于本示例。你必须手动编译 LibTorch——[此处] (https://gist.github.com/lasagnaphil/3e0099816837318e8e8bcab7edcfd5d9)提供了一组指导原则，不过具体做法可能因系统而异。

要运行代码，

```shell
mpirun -np {NUM-PROCS} ./dist-mnist
```
