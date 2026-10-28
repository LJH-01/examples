# 基础 VAE 示例

这是对 Kingma 和 Welling 的论文 [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) 的改进实现。
它使用 ReLU 和 adam 优化器，而不是 sigmoid 和 adagrad。这些改动使网络收敛快得多。

### 用法
安装所需的依赖：
```bash
pip install -r requirements.txt
```

要运行该示例，请执行：
```bash
python main.py
```

如果检测到硬件加速设备，该示例将在加速器上执行；否则将在 CPU 上运行。

要强制在 CPU 上执行，请使用 `--no-accel` 命令行参数：

```bash
python main.py --no-accel
```

main.py 脚本接受以下可选参数：

```bash
--batch-size            input batch size for training (default: 128)
--epochs                number of epochs to train (default: 10)
--no-accel              disables accelerator
--seed                  random seed (default: 1)
--log-interval	        how many batches to wait before logging training status
```
