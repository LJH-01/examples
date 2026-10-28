# minGPT-DDP

本目录的代码配合 https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html 上的教程，演示如何在 PyTorch 中使用分布式数据并行（DDP）训练一个类似 GPT 的模型。

标有星号（*）的文件改编自 minGPT 仓库（https://github.com/karpathy/minGPT）。

- [trainer.py](mingpt/trainer.py) 包含 Trainer 类，它在给定数据集上对模型执行分布式训练迭代。
- [model.py *](mingpt/model.py) 定义了模型架构。
- [char_dataset.py *](mingpt/char_dataset.py) 包含用于字符级数据集的 `Dataset` 类。
- [gpt2_train_cfg.yaml](mingpt/gpt2_train_cfg.yaml) 包含数据、模型、优化器和训练运行的配置。
- [main.py](mingpt/main.py) 是训练任务的入口。它设置 DDP 进程组、读取所有配置并运行训练任务。
- [slurm/](mingpt/slurm) 包含用于设置 AWS 集群的文件以及运行多节点训练的 slurm 脚本。
