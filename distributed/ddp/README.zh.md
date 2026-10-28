# 使用 PyTorch 的分布式数据并行（DDP）应用

本指南演示如何组织一个分布式模型训练应用，以便使用 `torchrun` 便捷地进行多节点启动。

---

## 前置条件

你应当熟悉：

- [PyTorch 基础](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [编写分布式应用](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [分布式模型训练](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

本教程使用 [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)（DDP）类进行数据并行训练：多个 worker 在不同的数据分片上训练同一个全局模型，计算本地梯度，并使用 AllReduce 对它们进行同步。在高性能计算（HPC）中，这被称为 _单程序多数据_（SPMD）。

---

## 应用的进程拓扑

一个分布式数据并行（DDP）应用可以在多个节点上执行，
每个节点可以包含多个 GPU 设备。
每个节点又可以运行 DDP 应用的多个副本，
每个副本在多个 GPU 上处理各自的模型。

设：
- _N_ = 节点数
- _G_ = 每个节点的 GPU 数
- _W_ = **World Size**（世界大小）= 进程总数
- _L_ = **Local World Size**（本地世界大小）= 每个节点的进程数

每个进程拥有：
- **本地 rank（Local rank）**：取值于 `[0, L-1]`
- **全局 rank（Global rank）**：取值于 `[0, W-1]`

**示例：**
如果你在 2 个节点上启动一个 DDP 应用，每个节点有 4 个 GPU，并且希望每个进程跨 2 个 GPU，则映射关系如下：

![ProcessMapping](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

虽然将进程映射到节点的方式有很多，但一条良好的经验法则是让一个进程对应一个 GPU。这使 DDP 应用拥有的并行读取流数量与 GPU 数量相同，在实践中可以在 I/O 与计算开销之间取得良好的平衡。在本教程的其余部分，我们假定应用遵循这一经验法则。

# 准备并启动 DDP 应用

无论 DDP 应用以何种方式启动，每个进程都需要一种机制来获知自己的全局 rank 和本地 rank。一旦获知这些信息，所有进程便创建一个 `ProcessGroup`，从而能够参与 AllReduce 等集合通信操作。

启动多个 DDP 进程并初始化创建 `ProcessGroup` 所需的所有取值，一种便捷的方式是使用 PyTorch 提供的 [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) 脚本。

---

## 示例应用

本示例基于 ["Hello, World" DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

应用会调用 `spmd_main` 入口函数：

```python
if __name__ == "__main__":
    spmd_main()
```

在 `spmd_main` 中，进程组使用 Accelerator API 初始化。其余的会合（rendezvous）信息来自 `torchrun` 设置的环境变量：

```python
def spmd_main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    acc = torch.accelerator.current_accelerator()
    vendor_backend = torch.distributed.get_default_backend_for_device(acc)
    torch.accelerator.set_device_index(rank)
    torch.distributed.init_process_group(backend=vendor_backend)

    demo_basic(rank)

    # Tear down the process group
    torch.distributed.destroy_process_group()
```

**要点：**
- 每个进程从环境变量中读取自己的 rank 和 world size。
- 初始化进程组以进行分布式通信。

训练函数 `demo_basic` 会在相应的 GPU 上初始化 DDP 模型：

```python
def demo_basic(rank):
    print(
        f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
        + f"world_size = {torch.distributed.get_world_size()}"
    )

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
```

---

## 启动应用

```sh
torchrun --nnodes=1 --nproc_per_node=8 example.py
```

---

## 输出示例

预期的输出：

```sh
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[238627] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '0', 'WORLD_SIZE': '8'}
[238630] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '3', 'WORLD_SIZE': '8'}
[238628] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '1', 'WORLD_SIZE': '8'}
[238634] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '7', 'WORLD_SIZE': '8'}
[238631] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '4', 'WORLD_SIZE': '8'}
[238632] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '5', 'WORLD_SIZE': '8'}
[238629] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '2', 'WORLD_SIZE': '8'}
[238633] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '6', 'WORLD_SIZE': '8'}
[238633] world_size = 8, rank = 6, backend=nccl
[238628] world_size = 8, rank = 1, backend=nccl
[238629] world_size = 8, rank = 2, backend=nccl
[238631] world_size = 8, rank = 4, backend=nccl
[238630] world_size = 8, rank = 3, backend=nccl
[238632] world_size = 8, rank = 5, backend=nccl
[238634] world_size = 8, rank = 7, backend=nccl
[238627] world_size = 8, rank = 0, backend=nccl
[238633] rank = 6, world_size = 8
[238628] rank = 1, world_size = 8
[238632] rank = 5, world_size = 8
[238634] rank = 7, world_size = 8
[238629] rank = 2, world_size = 8
[238630] rank = 3, world_size = 8
[238631] rank = 4, world_size = 8
[238627] rank = 0, world_size = 8
```

# 结论

作为分布式数据并行应用的作者，你的代码需要了解两类资源：计算节点以及每个节点内的 GPU。建立记录机制来跟踪 GPU 集合如何映射到应用的进程，这一过程可能繁琐且容易出错。我们希望，通过像本示例那样组织你的应用并使用 `torchrun`，可以显著简化分布式训练的搭建工作。
