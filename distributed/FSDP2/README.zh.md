## FSDP2
在 transformer 模型上运行 FSDP2：

```
cd distributed/FSDP2
pip install -r requirements.txt
torchrun --nproc_per_node 2 example.py
```
* 第一次运行时，它会创建一个 "checkpoints" 文件夹并在其中保存 state dict
* 第二次运行时，它会从之前的 checkpoint 加载

启用显式预取
```
torchrun --nproc_per_node 2 example.py --explicit-prefetch
```

启用混合精度
```
torchrun --nproc_per_node 2 example.py --mixed-precision
```

演示 DCP API
```
torchrun --nproc_per_node 2 example.py --dcp-api
```

## 确保你运行的是较新版本的 PyTorch：
参见 https://pytorch.org/get-started/locally/ 以安装至少 2.5 版本，理想情况下安装当前的 nightly 构建。
