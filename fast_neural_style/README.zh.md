# fast-neural-style :city_sunrise: :rocket:

本仓库包含一个艺术风格迁移算法的 PyTorch 实现。该算法可以将一幅图像的内容与另一幅图像的风格混合。例如，下面是一张以彩色玻璃画风格渲染的门拱照片。

该模型使用了 [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 中描述的方法，并结合了[实例归一化](https://arxiv.org/pdf/1607.08022.pdf)。README 中所示示例的已保存模型可从[此处](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)下载。

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## 环境要求

该程序使用 Python 编写，并使用了 [pytorch](http://pytorch.org/)、[scipy](https://www.scipy.org)。GPU 不是必需的，但它可以带来显著的加速，尤其是在训练新模型时。使用已保存的模型，可以在笔记本电脑或台式机上对常规尺寸的图像进行风格化处理。

## 用法

对图像进行风格化

```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --accel
```

- `--content-image`：你想要风格化的内容图像的路径。
- `--model`：用于对图像进行风格化的已保存模型（例如：`mosaic.pth`）
- `--output-image`：保存输出图像的路径。
- `--content-scale`：如果内存紧张，用于缩小内容图像的系数（例如：值为 2 会将内容图像的高度和宽度减半）
- `--accel`：使用加速器

训练模型

```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --accel
```

有若干命令行参数，重要的几个列在下面

- `--dataset`：训练数据集的路径，该路径应指向一个文件夹，其中还包含一个存放所有训练图像的文件夹。我使用了 COCO 2014 Training images 数据集 [80K/13GB] [（下载）](https://cocodataset.org/#download)。
- `--style-image`：风格图像的路径。
- `--save-model-dir`：保存训练所得模型的文件夹路径。
- `--accel`：使用加速器。

如果给出了 `--accel` 参数，pytorch 将搜索可用的硬件加速设备并尝试使用它。已知此示例可在 CUDA、MPS 和 XPU 设备上正常工作。

其他命令行参数请参阅 `neural_style/neural_style.py`。训练新模型时，你可能需要调整 `--content-weight` 和 `--style-weight` 的取值。上面展示的 mosaic 风格模型是使用 `--content-weight 1e5` 和 `--style-weight 1e10` 训练的。其余 3 个模型也使用相近数量级的权重参数训练，只是 `--style-weight` 略有变化（`5e10` 或 `1e11`）。

## 模型

下面所示示例的模型可从[此处](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)下载，或通过运行脚本 `download_saved_models.py` 下载。

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">		
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/rain-princess-cropped.jpg' height="174px">
  <img src='images/output-images/amber-rain-princess.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>
