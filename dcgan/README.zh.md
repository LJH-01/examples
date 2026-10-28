# 深度卷积生成对抗网络

本示例实现了论文 [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

该实现与 Torch 版本 [dcgan.torch](https://github.com/soumith/dcgan.torch) 非常接近

每经过 100 次训练迭代，生成模型的样本会以 `real_samples.png` 和 `fake_samples.png` 文件的形式写入磁盘。

每个 epoch 结束后，模型会保存为：`netG_epoch_%d.pth` 和 `netD_epoch_%d.pth`

## 下载数据集

你可以通过克隆[此仓库](https://github.com/fyu/lsun)并运行以下命令来下载 LSUN 数据集：

```
python download.py -c bedroom
```

## 用法

```
usage: main.py [-h] --dataset DATASET [--dataroot DATAROOT]
               [--workers WORKERS] [--batchSize BATCHSIZE]
               [--imageSize IMAGESIZE] [--nz NZ] [--ngf NGF] [--ndf NDF]
               [--niter NITER] [--lr LR] [--beta1 BETA1] [--dry-run]
               [--ngpu NGPU] [--netG NETG] [--netD NETD] [--outf OUTF]
               [--manualSeed MANUALSEED] [--classes CLASSES] [--accel]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF             number of generator filters
  --ndf NDF             number of discriminator filters
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --dry-run             check a single training cycle works
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed MANUALSEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --accel               enables accelerator
```
