# distributed-pytorch

DDP 系列教程（https://pytorch.org/tutorials/beginner/ddp_series_intro.html）的代码。

每个代码文件都在前一个文件的基础上扩展。该系列从单 GPU 上运行的非分布式脚本开始，逐步演进，最终以在 Slurm 集群上进行多节点训练收尾。

## 文件
* [single_gpu.py](single_gpu.py)：非分布式训练脚本

* [multigpu.py](multigpu.py)：单节点上的 DDP

* [multigpu_torchrun.py](multigpu_torchrun.py)：使用 Torchrun 在单节点上的 DDP

* [multinode.py](multinode.py)：使用 Torchrun（可选 Slurm）在多节点上的 DDP
    * [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md)：设置 AWS 集群的说明
    * [slurm/config.yaml.template](slurm/config.yaml.template)：设置 AWS 集群的配置
    * [slurm/sbatch_run.sh](slurm/sbatch_run.sh)：用于启动训练任务的 slurm 脚本
