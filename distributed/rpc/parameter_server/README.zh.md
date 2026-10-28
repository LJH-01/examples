### 基于 RPC 的分布式训练

这是一个基于 RPC 训练的基础示例，使用多个 trainer 远程训练托管在服务器上的模型。

要在本地运行该示例，请分别在独立的终端窗口中为服务器和你希望启动的每个 worker 运行以下命令：
`python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`。例如，对于 world size 为 2 的主节点，命令为 `python rpc_parameter_server.py --world_size=2 --rank=0`。随后可以在另一个窗口中用命令 `python rpc_parameter_server.py --world_size=2 --rank=1` 启动 trainer，这样便会开始以一个服务器和单个 trainer 进行训练。

请注意，出于演示目的，本示例仅支持 0 到 2 个 GPU，不过该模式可以扩展以使用更多 GPU。要配置 GPU 数量，请在训练命令中传入 `--num_gpus=N`。

你可以传入命令行参数 `--master_addr=<address>` 和 `master_port=PORT` 来指定主 worker 正在监听的地址:端口。所有 worker 都会在 worker 发现过程中联系主节点进行会合。默认情况下，`master_addr` 为 `localhost`，`master_port` 为 29500。
