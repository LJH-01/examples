# 使用高效亚像素卷积神经网络的超分辨率

本示例演示如何使用 Shi 等人论文[《Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network》](https://arxiv.org/abs/1609.05158)中描述的高效亚像素卷积层，在网络内部提升空间分辨率，以用于超分辨率等任务。

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--accel] [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --accel                use accelerator
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```

本示例在 [BSD300 数据集](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)上训练一个超分辨率网络，使用 200 张训练图像的裁剪块进行训练，并在 100 张测试图像的裁剪块上评估。每个 epoch 结束后会保存模型快照，文件名为 `model_epoch_<epoch_number>.pth`。

## 用法示例：

### 训练

```bash
python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001 --accel
```

### 超分辨率重建

```bash
python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png --accel
```
