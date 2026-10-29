# PyTorch Examples 全示例解析与学习路线

> 分析对象：`d:\project\python\examples`（pytorch/examples）
> 覆盖范围：39 个示例（20 个顶层示例 + 7 个 C++ 示例 + 12 个分布式示例），另含根目录脚本与 docs
> 文档目的：说明每个示例做什么、解决什么问题、适用什么场景，并给出有依赖依据的学习顺序
> 更新日期：2026-10-29

---

## 目录

- [一、仓库定位与阅读方式](#一仓库定位与阅读方式)
- [二、示例概览](#二示例概览)
- [三、详细说明](#三详细说明)
  - [3.1 入门与训练骨架](#31-入门与训练骨架)
  - [3.2 计算机视觉](#32-计算机视觉)
  - [3.3 生成模型](#33-生成模型)
  - [3.4 序列与文本](#34-序列与文本)
  - [3.5 图神经网络](#35-图神经网络)
  - [3.6 强化学习](#36-强化学习)
  - [3.7 分布式训练](#37-分布式训练)
  - [3.8 FX 与模型变换](#38-fx-与模型变换)
  - [3.9 C++ 前端（LibTorch）](#39-c-前端libtorch)
- [四、依赖关系与推荐学习顺序](#四依赖关系与推荐学习顺序)
- [五、分阶段学习目标与验收](#五分阶段学习目标与验收)
- [六、人工智能主要分支学习路线](#六人工智能主要分支学习路线)
  - [6.1 全景地图：AI 的分支与层级](#61-全景地图ai-的分支与层级)
  - [6.2 通用五层与时间预算](#62-通用五层与时间预算)
  - [6.3 分支一：计算机视觉（CV）](#63-分支一计算机视觉cv)
  - [6.4 分支二：自然语言处理（NLP / LLM）](#64-分支二自然语言处理nlp--llm)
  - [6.5 分支三：语音与音频](#65-分支三语音与音频)
  - [6.6 分支四：多模态与跨模态](#66-分支四多模态与跨模态)
  - [6.7 分支五：生成模型（横切底座）](#67-分支五生成模型横切底座)
  - [6.8 分支六：图学习与结构化数据](#68-分支六图学习与结构化数据)
  - [6.9 分支七：强化学习与决策](#69-分支七强化学习与决策)
  - [6.10 分支八：AI 系统与工程（横切）](#610-分支八ai-系统与工程横切)
  - [6.11 分支间的交叉与衔接](#611-分支间的交叉与衔接)
  - [6.12 三档时间排期](#612-三档时间排期)
  - [6.13 成果检验统一标准](#613-成果检验统一标准)
  - [6.14 与 examples 的映射](#614-与-examples-的映射)
  - [6.15 常见误区](#615-常见误区)
- [七、附录：环境与常见坑](#七附录环境与常见坑)

---

## 一、仓库定位与阅读方式

**仓库定位**：`pytorch/examples` 不是教程库，而是"可直接运行的最小完整程序"集合。每个示例都是一个自洽的训练/推理脚本，目标是"短、少依赖、彼此差异明显"。因此它适合用来**学工程写法**，不适合用来系统学理论。

**阅读每个示例的固定方法**：

1. 先看 `README.md`：确认运行命令、参数、预期输出
2. 再看入口文件：`main.py` / `main()`，找出数据、模型、损失、优化器、循环五个要素
3. 最后看模型定义：`model.py` 或同文件内的 `class xxx(nn.Module)`
4. 检查"有没有第二份实现"：C++ 示例、分布式示例常与 Python 版互为对照

**根目录工具**（不是示例，但会用到）：

| 文件 | 作用 |
|---|---|
| `run_python_examples.sh` | 批量跑 Python 示例的回归脚本 |
| `run_cpp_examples.sh` | 批量跑 C++ 示例 |
| `run_distributed_examples.sh` | 批量跑分布式示例 |
| `utils.sh` | 上述脚本共用的辅助函数 |
| `docs/` | Sphinx 文档工程，示例卡片登记在 `docs/source/index.rst` |

---

## 二、示例概览

难度标记：★ 入门（1 天内可跑通）｜★★ 进阶（需理解一个额外概念）｜★★★ 高级（需要配套知识或环境）｜★★★★ 专家（需要多卡/集群/多语言环境）

| 示例目录 | 主题 | 核心文件 | 难度 | 主要前置 |
|---|---|---|---|---|
| `regression/` | 线性回归 | `main.py` | ★ | 无 |
| `mnist/` | CNN 图像分类 | `main.py` | ★ | 无 |
| `mnist_rnn/` | RNN 图像分类 | `main.py` | ★★ | mnist |
| `mnist_hogwild/` | 多进程异步训练 | `main.py`、`train.py` | ★★★ | mnist |
| `mnist_forward_forward/` | 非反向传播学习 | `main.py` | ★★★ | mnist |
| `dcgan/` | 生成对抗网络 | `main.py` | ★★★ | mnist、vae（建议） |
| `vae/` | 变分自编码器 | `main.py` | ★★★ | mnist |
| `super_resolution/` | 超分辨率 | `main.py`、`model.py`、`data.py`、`dataset.py`、`super_resolve.py` | ★★ | mnist |
| `fast_neural_style/` | 风格迁移 | `neural_style/*.py` | ★★★ | imagenet（VGG 特征） |
| `siamese_network/` | 图像相似度 | `main.py` | ★★★ | mnist |
| `time_sequence_prediction/` | 时间序列预测 | `generate_sine_wave.py`、`train.py` | ★★ | 无 |
| `reinforcement_learning/` | 强化学习 | `reinforce.py`、`actor_critic.py` | ★★★ | mnist |
| `imagenet/` | 大规模分类训练 | `main.py`、`extract_ILSVRC.sh` | ★★★★ | mnist |
| `word_language_model/` | 语言建模 | `data.py`、`model.py`、`main.py`、`generate.py` | ★★★ | 无 |
| `language_translation/` | 机器翻译 | `main.py`、`src/model.py`、`src/data.py` | ★★★★ | word_language_model |
| `gcn/` | 图卷积网络 | `main.py` | ★★★ | mnist |
| `gat/` | 图注意力网络 | `main.py` | ★★★★ | gcn |
| `fx/` | 图变换与追踪 | 9 个独立脚本 | ★★★ | mnist |
| `fx/native_interpreter/` | 自定义运行时 | `interpreter.cpp`、`use_interpreter.py` | ★★★★ | fx |
| `legacy/snli/` | 文本蕴含分类 | `model.py`、`train.py`、`util.py` | ★★★ | word_language_model |
| `distributed/ddp/` | DDP 最小示例 | `example.py` | ★★★ | mnist |
| `distributed/ddp-tutorial-series/` | DDP 四步演进 | 4 个脚本 + `slurm/` | ★★★ | distributed/ddp |
| `distributed/FSDP/` | T5 摘要（FSDP1） | `T5_training.py`、`policies/*` | ★★★★ | word_language_model |
| `distributed/FSDP2/` | 分片训练 | `example.py`、`model.py`、`checkpoint.py` | ★★★★ | distributed/FSDP |
| `distributed/minGPT-ddp/` | GPT + DDP | `mingpt/*.py` | ★★★★ | word_language_model、ddp |
| `distributed/tensor_parallelism/` | 张量并行 | `llama2_model.py` 等 4 个 | ★★★★ | distributed/FSDP2 |
| `distributed/rpc/parameter_server/` | 参数服务器 | `rpc_parameter_server.py` | ★★★★ | mnist |
| `distributed/rpc/pipeline/` | 流水线并行 | `main.py` | ★★★★ | rpc/parameter_server |
| `distributed/rpc/rnn/` | RNN 模型并行 | `rnn.py`、`main.py` | ★★★★ | rpc/parameter_server |
| `distributed/rpc/rl/` | 多观察者强化学习 | `main.py` | ★★★★ | reinforcement_learning |
| `distributed/rpc/batch/` | 异步 RPC 批处理 | `parameter_server.py`、`reinforce.py` | ★★★★ | rpc/parameter_server |
| `distributed/rpc/ddp_rpc/` | DDP + RPC 混合 | `main.py` | ★★★★ | ddp、rpc |
| `cpp/regression/` | C++ 线性回归 | `regression.cpp` | ★★ | regression |
| `cpp/mnist/` | C++ MNIST | `mnist.cpp` | ★★ | mnist |
| `cpp/autograd/` | C++ 自动求导 | `autograd.cpp` | ★★★ | cpp/mnist |
| `cpp/dcgan/` | C++ DCGAN | `dcgan.cpp`、`display_samples.py` | ★★★★ | dcgan、cpp/mnist |
| `cpp/custom-dataset/` | C++ 自定义数据集 | `custom-data.cpp` | ★★★★ | cpp/mnist、OpenCV |
| `cpp/transfer-learning/` | C++ 迁移学习 | `convert.py` + C++ 源码 | ★★★★ | f/pth 导出、OpenCV |
| `cpp/distributed/` | C++ MPI 分布式 | `dist-mnist.cpp` | ★★★★ | cpp/mnist、MPI |

---

## 三、详细说明

### 3.1 入门与训练骨架

#### regression

- **作用**：用单层全连接网络拟合四次多项式 `y = 11.669x⁴ - 3.16023x³ + 2.19182x² - 3.81505x + 4.38219`。
- **解决的问题**：在没有任何真实数据集的干扰下，看清"训练到底在做什么"。没有 DataLoader、没有卷积，只有张量运算与梯度下降，便于排除干扰。
- **关键要点**：
  - `make_features()` 手工把一维 x 扩展成 `[x, x², x³, x⁴]`，是"特征工程"的最小隐喻
  - `poly_desc()` 把学到的权重还原成人类可读的多项式，用于判断"学得像不像"
  - `get_batch()` 每次采样随机 x，属于在线生成数据
- **适用场景**：新环境验证 PyTorch 是否装好；给新人讲清 loss/backward/step 三步。

#### mnist

- **作用**：两层卷积 + 两层全连接，在 MNIST 上做 10 分类。
- **解决的问题**：给出一个"标准到可以当模板"的监督学习脚本：参数解析、DataLoader、训练循环、验证、保存模型，一个不缺。
- **关键要点**：
  - `Net.forward` 中 `log_softmax` 配合 `NLLLoss`，与常见的 `softmax + CrossEntropyLoss` 等价，是理解损失函数实现差异的入口
  - `main()` 里 `--dry-run` 只跑一个 batch，是快速验证代码正确性的实用技巧
  - `transform=transforms.Compose([...ToTensor(), Normalize((0.1307,),(0.3081,))])` 展示了训练/推理必须一致的归一化
- **适用场景**：所有后续图像示例的骨架来源；改造为自有数据的起点。

#### mnist_rnn

- **作用**：把 28×28 的图片按行切成 28 步序列，用 RNN 做分类。
- **解决的问题**：同一份数据、同一任务，换一种建模视角会怎样。它建立了"序列建模"与"图像建模"之间的桥梁。
- **关键要点**：
  - `batch_first=True`、每步输入 `(batch, 28)`、隐藏状态跨步传递
  - 相比 CNN，缺少局部先验，所以收敛更慢、精度更低——这本身就是重要结论
- **适用场景**：准备学习 Transformer/序列模型前的热身；理解"归纳偏置"概念。

#### mnist_hogwild（进阶）

- **作用**：多个进程共享**同一份**模型参数，各自读数据、各自算梯度、无锁更新。
- **解决的问题**：多核 CPU 上训练小模型时，Python 单进程利用率低。Hogwild 用"允许梯度冲突"的代价换取吞吐。
- **关键要点**：
  - `model.share_memory()` 把参数放进共享内存
  - `torch.multiprocessing.Process` 启动 N 个 `train(rank, ...)`，每个进程独立 DataLoader
  - 无锁更新会导致梯度被覆盖，但实践中对稀疏梯度/小模型影响有限
- **适用场景**：CPU 集群上的轻量模型；理解"分布式训练不一定需要同步"。

#### mnist_forward_forward（进阶）

- **作用**：实现 Hinton 提出的 Forward-Forward 算法，用两次前向传播替代反向传播。
- **解决的问题**：反向传播需要保存整条计算图，显存开销大、生物学上不合理。该算法用"局部正/负 goodness"逐层训练。
- **关键要点**：
  - `overlay_y_on_x()` 把标签叠加到图像边上，构造"正样本"
  - `get_y_neg()` 换错标签构造"负样本"
  - 每层独立训练，用 `Layer`（继承 `nn.Linear`）计算 goodness
- **适用场景**：研究替代学习范式；理解"局部损失"与"全局反传"的区别。

---

### 3.2 计算机视觉

#### imagenet

- **作用**：在 ImageNet 上训练 ResNet/AlexNet/VGG 等主流架构，支持单机多卡与多机多卡。
- **解决的问题**：真实规模训练工程问题——大数据吞吐、混合精度、多进程、断点续训、top-k 评测。
- **关键要点**：
  - `main()` 解析参数并做 `mp.spawn`；`main_worker()` 才是每个进程真正执行的函数
  - `train()` 内含 AMP（`torch.amp.autocast` + `GradScaler`）与学习率按 epoch 衰减
  - `accuracy(output, target, topk=(1,5))` 是标准 top-k 实现，可改造成检索指标
  - `save_checkpoint()` 保存 `state_dict / optimizer / epoch / best_acc1`，是断点续训的标准结构
  - `--dummy` 用假数据做速度基准，`--pretrained` 加载预训练权重
- **适用场景**：训练自有骨干网络；学习 AMP 与 DDP 启动方式；提取中间层特征做下游任务。

#### super_resolution

- **作用**：用亚像素卷积（PixelShuffle）把低分辨率图像放大 2/3/4 倍，在 BSD300 上训练与评估。
- **解决的问题**：上采样不等于"变大"——普通插值会糊，亚像素卷积让网络在低分辨率空间算特征、最后一次性重排成高分辨率。
- **关键要点**：
  - `model.py` 的 `Net`：卷积提特征 → `PixelShuffle(upscale_factor)` → 输出
  - `data.py` 的 `get_training_set/get_test_set` 自动下载 BSD300 并做裁剪配对
  - `dataset.py` 的 `DatasetFromFolder` 是自定义数据集的规范写法
  - `super_resolve.py` 展示"加载权重 → 单图推理 → 保存"的完整推理脚本范式
- **适用场景**：商品图分辨率不足时的预处理；学习图像到图像任务与多输入（低清/高清）配对数据集。

#### fast_neural_style

- **作用**：训练一个前馈网络完成艺术风格迁移，推理时一次前向即可出图。
- **解决的问题**：原始的风格迁移（Gatys 方法）每张图都要迭代优化，太慢。该实现用感知损失训练一个"变换网络"，把优化搬到训练阶段。
- **关键要点**：
  - `transformer_net.py`：`ConvLayer` → 若干 `ResidualBlock` → `UpsampleConvLayer` 逐级放大（避免棋盘格伪影）
  - `vgg.py`：`Vgg16` 提供 `slice1_2`…`slice5_3`，用于取多层特征算内容损失与风格损失
  - `neural_style.py` 分 `train` 与 `eval` 两个子命令，是"一个脚本两种角色"的典型写法
  - `--content-weight 1e5` / `--style-weight 1e10` 量级差异极大，是感知损失的常见现象
- **适用场景**：商品图风格/背景归一化；学习感知损失与前馈式图像生成。

#### siamese_network

- **作用**：孪生网络判断两张图是否"相同"，在 MNIST 上演示图像相似度估计。
- **解决的问题**：不是所有识别任务都是"多分类"。当类别会持续新增、或需要给出"相似度分数"时，度量学习比分类更合适。
- **关键要点**：
  - `SiameseNetwork`：两个输入共享同一个 ResNet18（同一份权重），特征拼接后经全连接 + Sigmoid 输出相似概率
  - `APP_MATCHER`：自定义 Dataset，`group_examples()` 按类别分组，`__getitem__` 返回"同类对"或"异类对"
  - 用 `BCELoss` 而非 TripletLoss，因为 MNIST 太简单——这个取舍本身就是知识点
  - 局限：输出的是"成对分数"，无法离线建库检索，需要改造成 embedding + 余弦
- **适用场景**：人脸/商品同款/去重；学习正负样本构造与度量学习入门。

---

### 3.3 生成模型

#### vae

- **作用**：实现变分自编码器（VAE），在 MNIST 上做重构与生成。
- **解决的问题**：普通自编码器的潜空间是"散乱"的，无法采样生成新样本。VAE 用 KL 正则把潜变量约束到标准正态分布。
- **关键要点**：
  - `VAE.encode` 输出 `mu` 与 `logvar`，`reparametrize` 用 `mu + eps * std` 实现可导采样
  - `loss_function()` = 重构损失（BCE）+ KL 散度，两者权重需要平衡
  - 用 ReLU + Adam 替代原始论文的 sigmoid + adagrad，收敛更快
- **适用场景**：数据增广（生成同款式变体）、embedding 分布诊断、学习潜变量模型。

#### dcgan

- **作用**：深度卷积生成对抗网络，生成与训练集同分布的图像（默认 LSUN bedroom）。
- **解决的问题**：如何在没有显式似然函数的情况下学习数据分布。GAN 用"生成器 vs 判别器"的对抗替代重构损失。
- **关键要点**：
  - 参数约定已成为事实标准：`nz=100`（潜向量维度）、`ngf/ndf=64`（生成器/判别器通道基数）、`lr=0.0002`、`beta1=0.5`
  - 每 100 次迭代保存 `real_samples.png` / `fake_samples.png`，每 epoch 保存 `netG_epoch_%d.pth` / `netD_epoch_%d.pth`
  - 支持 `lsun / cifar10 / mnist / folder / lfw / fake` 多种数据源，`--dry-run` 可先验证链路
- **适用场景**：长尾类目样本补充；理解生成模型训练的脆弱性（模式崩溃、震荡）。

---

### 3.4 序列与文本

#### time_sequence_prediction

- **作用**：用两个 `LSTMCell` 学习从不同相位开始的正弦波，并预测未来值。
- **解决的问题**：经典的"序列到序列"预测问题，且完全可控（数据是合成的），便于观察模型是否真的学到周期规律。
- **关键要点**：
  - `generate_sine_wave.py` 生成并保存数据，`train.py` 中 `Sequence` 类逐时间步喂入
  - 用 `LSTMCell` 而非 `LSTM`，是为了显式控制每一步的隐状态传递
  - 结果图中实线是已知信号、虚线是预测，可直接目视判断拟合质量
- **适用场景**：销量/流量预测入门；理解隐状态与序列长度。

#### word_language_model

- **作用**：在 Wikitext-2 上训练词级语言模型，支持 `RNN_TANH / RNN_RELU / LSTM / GRU / Transformer` 五种结构，并可采样生成文本。
- **解决的问题**：NLP 的基础任务——给定前文预测下一个词。它同时是理解 Transformer 的最短路径。
- **关键要点**：
  - `data.py`：`Dictionary` 管理词表（含 `<eos>`、未知词），`Corpus` 把文本转成 id 序列
  - `model.py`：`RNNModel`（含 `--tied` 权重共享）、`PositionalEncoding`、`TransformerModel`（继承 `nn.Transformer`，含因果掩码）
  - `main.py`：`batchify()` 把语料切成列式 batch，`get_batch()` 沿序列切片，`evaluate()` 计算 perplexity
  - `generate.py`：加载 checkpoint 做自回归采样，`--temperature` 控制多样性
  - `--onnx-export` 演示模型导出
- **适用场景**：商品标题建模、类目预测的预训练、所有 LLM 学习的起点。

#### language_translation

- **作用**：用 Transformer 做德译英（可换任意语言对），基于 "Attention is All You Need"。
- **解决的问题**：序列到序列任务中，编码器与解码器如何通过掩码与注意力对齐；训练（teacher forcing）与推理（自回归）为何必须分开实现。
- **关键要点**：
  - `src/model.py` 的 `Translator`：编码器 + 解码器 + 输出投影
  - `src/data.py`：`get_data()` 用 spacy 分词构建词表，`create_mask()` / `generate_square_subsequent_mask()` 生成填充掩码与因果掩码
  - `main.py` 的 `greedy_decode()` 是推理路径，`train()` 用 teacher forcing，`validate()` 计算 loss
  - 依赖 spacy 语言包：`python3 -m spacy download de/en`
- **适用场景**：跨境商品标题翻译；理解 encoder-decoder 与掩码机制。

#### legacy/snli

- **作用**：在 SNLI 语料上做自然语言推理（蕴含/矛盾/中立三分类）。
- **解决的问题**：句子对分类任务——两句输入如何编码与交互；同时展示预训练词向量（GloVe）的加载与冻结。
- **关键要点**：
  - `model.py`：`Encoder` 用 LSTM 编码句子，`SNLIClassifier` 拼接两句表示后分类，`Bottle` 是参数形状处理的辅助类
  - `util.py`：`get_args()` 集中管理参数，`makedirs()` 做目录兜底
  - 标注为 `legacy`，说明它依赖旧版 torchtext 接口，是"能跑但要改"的历史代码
- **适用场景**：文本分类/句子对任务的模板；理解接口演进带来的维护成本。

---

### 3.5 图神经网络

#### gcn

- **作用**：在 Cora 引文网络上做半监督节点分类（论文 *Semi-Supervised Classification with GCN*）。
- **解决的问题**：当样本之间有关系（引用、同款、同店）时，如何让邻居的信息参与自身表示的学习。
- **关键要点**：
  - `GraphConv`：`A_hat = D^-1/2 (A+I) D^-1/2`，再与特征相乘、线性变换
  - `load_cora()` 自动下载并预处理 Cora（特征、邻接、标签、掩码）
  - 半监督体现在：只有少量节点有标签（`mask_train`），其余用于验证/测试
  - 报告中 200 epoch 约 82.5% 准确率
- **适用场景**：商品关系图上的类目传播；同款关系增强召回。

#### gat

- **作用**：图注意力网络，用多头注意力为邻居分配不同权重。
- **解决的问题**：GCN 对所有邻居一视同仁（权重由度决定）。GAT 让模型自己学"该听谁的"。
- **关键要点**：
  - `GraphAttentionLayer`：`LeakyReLU` 作用于注意力系数，`ELU` 作用于隐藏层输出
  - 第一层 8 个头 × 8 维 = 64 维，第二层单头输出类别数，接 log-softmax 配合 NLLLoss
  - 教学实现使用**稠密**邻接矩阵，非稀疏；不影响精度但影响显存与速度
- **适用场景**：邻居重要性差异明显的场景（如同款关系强弱不一）。

---

### 3.6 强化学习

#### reinforcement_learning

- **作用**：在 OpenAI Gym 的 CartPole 上实现两种策略梯度算法：`reinforce.py`（REINFORCE）与 `actor_critic.py`（Actor-Critic）。
- **解决的问题**：没有标签、只有延迟奖励时如何优化策略。两个文件正好构成"高方差 → 加基线降方差"的对照实验。
- **关键要点**：
  - `Policy` 网络输出动作概率分布；`select_action()` 从分布采样
  - `finish_episode()` 用累计回报计算损失：`-log_prob * G`
  - Actor-Critic 增加价值网络作为 baseline，降低梯度方差
  - 目标是演示算法而非刷分，CartPole 是最小可验证环境
- **适用场景**：排序策略优化、搜索/推荐中的序列决策入门。

---

### 3.7 分布式训练

#### distributed/ddp

- **作用**：DDP 的最小可运行示例，演示进程组初始化与 `torchrun` 启动。
- **解决的问题**：多卡训练的样板代码——rank 怎么来、进程组怎么建、模型怎么包。
- **关键要点**：
  - `spmd_main()` 从环境变量读取 `MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE / LOCAL_RANK`
  - `torch.distributed.init_process_group(backend=...)` 建立通信组，结束后 `destroy_process_group()`
  - `demo_basic()` 里 `DDP(model, device_ids=[rank])` 包模型，其余训练代码与单卡一致
  - 启动：`torchrun --nnodes=1 --nproc_per_node=8 example.py`
- **适用场景**：任何多卡训练任务的起点模板。

#### distributed/ddp-tutorial-series

- **作用**：四个脚本逐级演进：`single_gpu.py` → `multigpu.py` → `multigpu_torchrun.py` → `multinode.py`。
- **解决的问题**：直接看多机代码容易迷失。该系列把"加分布式"拆成三步可对比的增量。
- **关键要点**：
  - 每步只改少量代码，diff 就是知识点（Sampler 分片、`local_rank` 来源、启动方式）
  - `datautils.py` 抽出公共的 DataLoader 构造
  - `slurm/config.yaml.template` 与 `sbatch_run.sh` 提供 AWS 集群启动方式
- **适用场景**：从单卡平滑迁移到多机；给团队做分布式培训材料。

#### distributed/FSDP

- **作用**：用 FSDP1 微调 T5 做文本摘要，含混合精度、激活检查点、模型分片策略与 checkpoint 管理。
- **解决的问题**：模型参数 + 梯度 + 优化器状态超出单卡显存时，如何把它们切到多卡上，同时保持训练逻辑不变。
- **关键要点**：
  - `T5_training.py` 是主流程；`configs/fsdp.py`、`configs/training.py` 集中配置
  - `policies/wrapping.py`（分层包装）、`mixed_precision.py`、`activation_checkpointing_functions.py` 三件套
  - `model_checkpointing/checkpoint_handler.py` 处理分片权重的保存与加载
  - 注意：FSDP1 已弃用，官方推荐 FSDP2
- **适用场景**：大模型微调；理解分片、混合精度、激活重计算的组合。

#### distributed/FSDP2

- **作用**：FSDP2 的最小可运行示例，训练一个小型 transformer，支持显式预取、混合精度、DCP API。
- **解决的问题**：FSDP1 的 API 复杂、状态管理混乱。FSDP2 用 `fully_shard` 逐模块分片，更贴近 `nn.Module` 的使用习惯。
- **关键要点**：
  - `example.py` 主流程；`model.py` 定义 transformer；`checkpoint.py` 演示 DCP（分布式 checkpoint）
  - 首次运行创建 `checkpoints/` 并写 state dict，第二次运行从上次 checkpoint 加载
  - 开关项清晰：`--explicit-prefetch`、`--mixed-precision`、`--dcp-api`
  - 要求 PyTorch ≥ 2.5
- **适用场景**：显存受限时的首选方案；替代 FSDP1 的新写法。

#### distributed/minGPT-ddp

- **作用**：把 Karpathy 的 minGPT 用 DDP 训练，代码结构接近真实大模型工程。
- **解决的问题**：GPT 类模型如何组织代码（配置、数据集、训练器、入口分离），以及如何在多卡上跑。
- **关键要点**：
  - `model.py`：`GPT` / `MultiheadAttentionLayer` / `Block` / `EmbeddingStem`，带 `*` 的文件改编自 minGPT
  - `trainer.py`：`Trainer` 类封装训练迭代与 checkpoint
  - `char_dataset.py`：`CharDataset`，字符级数据集
  - `main.py`：设置 DDP 进程组、读配置、启动训练；`gpt2_train_cfg.yaml` 是唯一配置入口
- **适用场景**：从"能跑 GPT"到"能改 GPT"；大模型训练流程的最小样板。

#### distributed/tensor_parallelism

- **作用**：三个层次递进的示例——纯张量并行、带序列并行的张量并行、FSDP + TP 的 Llama2 端到端。
- **解决的问题**：单层就放不下的大模型（如 Llama2）该如何按列/行切分权重，以及切分后通信如何组织。
- **关键要点**：
  - `tensor_parallel_example.py` / `sequence_parallel_example.py`：`ToyModel` 上演示 `parallelize_module`
  - `llama2_model.py`：`RMSNorm` / `Attention`（含 RoPE、GQA）/ `FeedForward`（SwiGLU）/ `TransformerBlock`
  - `fsdp_tp_example.py`：FSDP 做参数分片、TP 做层内切分，二者组合
- **适用场景**：训练/推理超大模型；理解 Megatron 风格并行的最小实现。

#### distributed/rpc 系列

| 示例 | 作用 | 解决的问题 | 关键要点 |
|---|---|---|---|
| `parameter_server/` | 多 trainer 远程训练一个服务器上的模型 | 模型太大放不进单机，或需要集中式参数更新 | `rpc_parameter_server.py`，`--world_size/--rank/--num_gpus`，`master_addr/master_port` 会合 |
| `pipeline/` | 把 ResNet50 切到两个 worker 上做流水线并行 | 单卡放不下，又想利用多卡的空闲时间 | 输入切成 micro-batch 流水送入，仅并行化前向 |
| `rnn/` | RNN 的不同层放到不同 worker | 模型按层切分（模型并行） | `rnn.py` 定义可远程调用的模块，`main.py` 编排 |
| `rl/` | 多观察者 + 单智能体的强化学习 | 多个环境并行采样，策略集中更新 | `rpc.rpc_sync` 收集观测，Actor 集中推理 |
| `batch/` | `@rpc.functions.async_execution` 的两种用法 | 一次 RPC 处理一批请求，提升吞吐 | `parameter_server.py`（批量更新）、`reinforce.py`（批量观测） |
| `ddp_rpc/` | DDP + RPC 混合：稠密部分 DDP、稀疏部分在参数服务器 | 推荐系统式模型：EmbeddingBag 太大，不适合每卡复制 | `nn.Linear` 走 DDP，`nn.EmbeddingBag` 放参数服务器，反向时分别更新 |

- **适用场景**：推荐/搜索的粗排模型、超大 embedding 表、模型并行推理。

---

### 3.8 FX 与模型变换

`fx/` 不是单一示例，而是 9 个互相独立的脚本，每个演示一种图操作能力。建议按下列顺序读：

| 脚本 | 作用 | 解决的问题 | 关键 API |
|---|---|---|---|
| `proxy_based_graph_creation.py` | 不追踪、直接用 Proxy 搭图 | 想程序化生成计算图（如自动搜索、图融合） | `Proxy`、`Graph`、`GraphModule` |
| `replace_op.py` | 把图中某个算子替换成另一个 | 后端不支持某算子，需要等价改写 | `replace_all_uses_with`、`recompile` |
| `subgraph_rewriter_basic_use.py` | 匹配子图并整体替换 | 把"conv+bn+relu"合成一个融合算子 | `replace_pattern` |
| `module_tracer.py` | 记录每个操作来自哪个子模块 | 定位问题层、按模块做量化 | `Tracer` 子类、qualified name |
| `custom_tracer.py` | 自定义追踪行为（下沉 ReLU、打标签） | 厂商后端需要在追踪阶段改变图结构 | `Tracer` 覆写 `call_module/call_function` |
| `inline_function.py` | 把函数调用在图中内联展开 | 默认追踪把 `nn.Module` 记为 `call_module`，不展开内部 | `map_arg`、手动插入节点 |
| `primitive_library.py` | 定义"复合算子"库并自动内联 | 想以更高抽象层操作图，又不丢失底层可见性 | `InliningTracer` |
| `profiling_tracer.py` | 让 profiler 区间出现在 FX 图中 | 想按图节点归因耗时 | `torch.ops.profiler._record_function_enter/exit` |
| `wrap_output_dynamically.py` | 运行时按参数动态改输出 | 同一个图需要支持多种输出后处理 | `Proxy`、动态插入 `output` 节点 |

#### fx/native_interpreter

- **作用**：把 PyTorch 模块"下沉"成自定义指令序列，由 C++ 写的解释器执行，并通过 TorchScript 自定义类暴露给 Python。
- **解决的问题**：模型部署时希望脱离 Python 运行时、或接入自有后端（NPU/专用芯片）。该示例给出"lowering → 解释器 → 自定义类"的完整链路。
- **关键要点**：
  - `interpreter.cpp`：只支持 `add` / `mul` 两个逐元素算子，仅接受 `Tensor`
  - `use_interpreter.py`：`MyElementwiseModule` 被 lowering 成指令列表，`torch.classes.load_library` 加载 `.so`
  - 构建依赖 CMake 与 `torch.utils.cmake_prefix_path`
- **适用场景**：自研推理引擎的前端对接；理解"图 → IR → 执行器"的分层。

---

### 3.9 C++ 前端（LibTorch）

C++ 示例共享同一套构建方式：`mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. && make`。

| 示例 | 作用 | 解决的问题 | 关键要点 |
|---|---|---|---|
| `cpp/regression/` | C++ 版多项式拟合 | 用最小程序验证 LibTorch 环境 | 单层 `nn::Linear`，SGD，输出学到的系数 |
| `cpp/mnist/` | C++ 版 MNIST 训练 | 在 C++ 中完成数据加载、训练、测试全流程 | `torch::data::make_data_loader`、`torch::save` |
| `cpp/autograd/` | 三种自动求导用法 | C++ 侧如何手写反向、求高阶梯度、自定义 Function | 基本运算、`torch::autograd::Function` 子类、grad of grad |
| `cpp/dcgan/` | C++ 版 DCGAN | 在 C++ 中实现生成对抗训练 | `--epochs` 控制轮数，`display_samples.py` 可视化 `.pt` 样本 |
| `cpp/custom-dataset/` | 用 OpenCV 读自定义图像数据集并训练 | 真实图像不是张量，需要解码、缩放、归一化 | Caltech 101，`cv::imread`，若链接失败需换 `libtorch-cxx11-abi-shared-with-deps` |
| `cpp/transfer-learning/` | 冻结 ResNet18 骨干，只训线性头 | 小数据集上从头训练会过拟合 | `convert.py` 去掉最后一层并导出 TorchScript，`classify` 做预测 |
| `cpp/distributed/` | 用 MPI 做数据并行 | 无 NVIDIA 环境也想多机训练 | `mpirun -np N ./dist-mnist`，LibTorch 需自行编译以包含 MPI 头文件 |

---

## 四、依赖关系与推荐学习顺序

### 4.1 依赖关系

```
regression ──┐
             ├─→ mnist ─┬─→ mnist_rnn
             │          ├─→ mnist_hogwild
             │          ├─→ mnist_forward_forward
             │          ├─→ vae ─→ dcgan
             │          ├─→ super_resolution
             │          ├─→ siamese_network
             │          ├─→ reinforcement_learning ─→ rpc/rl
             │          ├─→ gcn ─→ gat
             │          └─→ imagenet ─→ fast_neural_style
             │
             ├─→ time_sequence_prediction
             ├─→ word_language_model ─┬─→ language_translation
             │                        ├─→ legacy/snli
             │                        ├─→ minGPT-ddp
             │                        └─→ FSDP（T5）
             │
             ├─→ distributed/ddp ─→ ddp-tutorial-series ─┬─→ minGPT-ddp
             │                                           └─→ FSDP2 ─→ tensor_parallelism
             │
             └─→ fx ─→ fx/native_interpreter
                       
cpp/regression ─→ cpp/mnist ─┬─→ cpp/autograd
                             ├─→ cpp/dcgan
                             ├─→ cpp/custom-dataset ─→ cpp/transfer-learning
                             └─→ cpp/distributed
```

### 4.2 排序原则

1. **概念前置**：后一个示例至少复用前一个的一个核心概念（如 `siamese_network` 复用 `mnist` 的训练循环与 `imagenet` 的骨干）。
2. **反馈周期优先**：能在几分钟内跑出结果的排前面，避免一开始就卡在数据下载或环境配置上。
3. **环境依赖后置**：需要多卡、MPI、OpenCV、Sphinx 的示例集中在后期，避免早期受环境阻塞。
4. **同族成组**：同一主题的示例连续学，减少上下文切换成本（如 6 个 rpc 示例连排）。
5. **难度阶梯**：严格递增，但允许在每阶段末尾放一个"跳一跳"的示例作为挑战。

### 4.3 推荐顺序（9 个阶段，约 14–16 周）

| 阶段 | 周次 | 示例顺序 | 为什么排在这里 |
|---|---|---|---|
| A 训练骨架 | 1 | `regression` → `mnist` | 无外部依赖，几秒出结果；建立"数据-模型-损失-循环"四要素认知 |
| B 视觉基础 | 2–3 | `imagenet` → `siamese_network` → `super_resolution` | 先有骨干（imagenet）才能谈特征复用；siamese 把分类改造成相似度；super_resolution 补图像到图像范式 |
| C 序列建模 | 3–4 | `mnist_rnn` → `time_sequence_prediction` | 从"图像当序列"过渡到"真正的序列预测"，难度平滑 |
| D 文本与 Transformer | 4–6 | `word_language_model` → `language_translation` → `legacy/snli` | 语言模型是 Transformer 最短路径；翻译需要 LM 的词表与 batching 基础；snli 收尾做句子对任务 |
| E 图与强化学习 | 6–7 | `gcn` → `gat`；`reinforcement_learning` | 两者都建立在 `mnist` 的训练循环上，概念独立，可并行推进 |
| F 生成模型 | 7–8 | `vae` → `dcgan` → `mnist_forward_forward` | 先学有重构损失的 VAE，再学无显式似然的 GAN；forward-forward 是对反向传播的反思，放在最后 |
| G 分布式训练 | 9–11 | `distributed/ddp` → `ddp-tutorial-series` → `mnist_hogwild` → `minGPT-ddp` → `FSDP` → `FSDP2` → `tensor_parallelism` → `rpc/*` | 必须已有能跑的模型（阶段 B/D 的产物）；从"最小 DDP"到"分片"再到"张量并行"，最后才碰 RPC 这类架构级并行 |
| H 图变换与部署 | 12–13 | `fx`（9 个脚本按表序） → `fx/native_interpreter` | 需要已理解模型结构（阶段 B）；部署是"模型跑通之后"的问题 |
| I C++ 前端 | 14–15 | `cpp/regression` → `cpp/mnist` → `cpp/autograd` → `cpp/dcgan` → `cpp/custom-dataset` → `cpp/transfer-learning` → `cpp/distributed` | 从"验证环境"到"复刻任务"再到"接真实数据/多机"；学习曲线由语言熟悉度决定，放最后最稳妥 |

**顺序中的三处刻意安排**：

- `fast_neural_style` 放在阶段 B 末而非阶段 F：它本质是"用 VGG 特征做损失"，依赖 `imagenet` 的预训练权重与特征提取认知，与生成模型关系反而较远。
- `minGPT-ddp` 放在 `ddp` 之后、`FSDP` 之前：先用小模型把 DDP 打通，再上分片；否则一上来就遇到显存/配置问题会打击信心。
- 6 个 `rpc` 示例整体置于阶段 G 末尾：它们解决的是"架构级并行"，没有前序的 DDP/FSDP 基础很难判断何时该用。

---

## 五、分阶段学习目标与验收

| 阶段 | 学习目标 | 验收方式 |
|---|---|---|
| A 训练骨架 | 能不看代码写出完整训练循环；能解释 `log_softmax + NLLLoss` 与 `CrossEntropyLoss` 的等价性 | 用 `regression` 的骨架改写一个二分类任务并收敛 |
| B 视觉基础 | 理解迁移学习、特征复用、度量学习三件事；能手改 `SiameseNetwork` 输出 embedding | 把 `siamese_network` 改成 embedding + 余弦，指标不下降 |
| C 序列建模 | 理解隐状态、序列长度、批次组织 | 改 `time_sequence_prediction` 预测复合正弦波（两个频率叠加） |
| D 文本与 Transformer | 能画出 `TransformerModel` 数据流；理解 mask 的作用 | 用 `word_language_model` 在自有语料训练并生成合理文本；手写 attention 通过形状校验 |
| E 图与强化学习 | 理解邻接矩阵归一化与注意力聚合；理解策略梯度与 baseline | 在 `gcn` 上加一层并比较指标；把 `reinforce.py` 改成带 baseline 版本并比较方差 |
| F 生成模型 | 理解重构损失、对抗损失、KL 正则的差异 | 用 `vae` 生成样本并做增广实验；记录 GAN 训练失败的典型现象 |
| G 分布式训练 | 能独立把单卡脚本改成多卡；能判断该用 DDP/FSDP/TP/RPC 中的哪种 | 4 卡跑通 `ddp-tutorial-series/multinode.py`；画出 FSDP2 与 DDP 的显存对比 |
| H 图变换与部署 | 理解 symbolic trace 的边界；能用 fx 做一次算子替换 | 用 `replace_pattern` 融合一个子图并验证数值一致；编译跑通 `interpreter.cpp` |
| I C++ 前端 | 能用 LibTorch 复刻一个 Python 示例 | 用 `cpp/mnist` 改造成自有数据集分类器 |

---

## 六、人工智能主要分支学习路线

> 本章与前三章的分工：第一至五章回答"这个仓库里有什么、怎么按代码学"；本章回答"整个 AI 领域有哪些分支、每个分支怎么走、彼此怎么接上"。
> 6.14 节给出两者的映射表，可以直接把本章的知识点落到仓库里的具体示例上。

### 6.1 全景地图：AI 的分支与层级

AI 的学习不是"选一个方向猛冲"，而是"先建公共底座，再选分支纵深，最后做交叉"。整体分为五层：

```
L5 系统与落地       部署 / 推理优化 / MLOps / 评测与数据飞轮
                    ↑ 所有分支最终都要落到这一层
L4 交叉融合         多模态 / 具身智能 / AI4Science / 决策智能
                    ↑ 两个及以上分支能力的组合
L3 分支专精         CV   NLP  语音  图学习  强化学习  生成模型
                    ↑ 每个分支有自己的问题范式与评价体系
L2 深度学习通用     反向传播 / 优化器 / 正则化 / 训练工程 / 表示学习
                    ↑ 所有分支共享的方法论
L1 机器学习基础     概率统计 / 线性模型 / 树模型 / 评估方法 / 特征工程
                    ↑ 决定你能否判断"该不该上深度学习"
L0 数学与编程       线性代数 / 概率 / 微积分 / Python / 数据结构 / 工程习惯
```

**八个分支的核心对象与定位**：

| 分支 | 核心对象 | 代表任务 | 与同款识别项目的关系 | 主要依赖 |
|---|---|---|---|---|
| 计算机视觉 CV | 像素、图像、视频 | 分类、检测、分割、检索、生成 | 直接支撑（图搜图） | L0–L2 |
| 自然语言处理 NLP | 词、句、篇章 | 分类、抽取、翻译、生成、问答 | 标题/类目/属性信号 | L0–L2 |
| 语音与音频 | 波形、频谱 | 识别、合成、声纹、事件检测 | 视频商品理解（选做） | L1–L2、CV/NLP |
| 多模态 | 图文音视频联合表示 | 图文检索、VQA、文档理解 | 图文同款、以文搜图 | CV + NLP |
| 生成模型 | 数据分布 | 采样、条件生成、编辑 | 长尾增广、白底图生成 | L2 + CV/NLP |
| 图学习 | 节点、边、子图 | 节点分类、链接预测、推荐 | 同款/搭配关系传播 | L2、NLP（图嵌入） |
| 强化学习 | 状态、动作、奖励 | 控制、博弈、排序、对齐 | 排序策略（远期） | L1、L2 |
| AI 系统与工程 | 算力、显存、延迟 | 训练加速、推理部署、监控 | 上线必备 | L0–L2 |

### 6.2 通用五层与时间预算

| 层级 | 核心内容 | 建议投入 | 达标标准 |
|---|---|---|---|
| L0 数学与编程 | 线代（矩阵乘法、特征值）、概率（分布、期望、贝叶斯）、微积分（链式法则）、Python 工程（模块、日志、测试） | 2–3 周（可与 L1 并行） | 能手推 2 层网络的梯度；能写可复用的训练脚本 |
| L1 机器学习 | 偏差-方差、正则化、逻辑回归、树模型、交叉验证、指标设计 | 3–4 周 | 能用 sklearn 完成一次完整建模并解释每个指标 |
| L2 深度学习通用 | 反向传播、优化器（SGD/Adam）、归一化、初始化、过拟合诊断、混合精度、数据管道 | 4–5 周 | 能从零实现 MLP 训练；能诊断"不收敛"的 5 类原因 |
| L3 分支专精 | 见 6.3–6.10，选 1 个主分支 + 1 个辅分支 | 每分支 6–10 周 | 在该分支完成一个可演示、有指标的项目 |
| L4 交叉融合 | 多模态、跨任务迁移、决策系统 | 4–6 周 | 能把两个分支的模型/特征接到同一链路 |
| L5 系统与落地 | 分布式训练、推理优化、服务化、评测体系 | 4–6 周 | 端到端系统上线并有监控指标 |

**总时长参考**：L0–L2 约 10–12 周（每周 8–10 小时）；主分支到可交付约再需 10–12 周；达到"能独立负责一个方向"通常需要 9–12 个月。不要试图并行推进三个以上分支。

### 6.3 分支一：计算机视觉（CV）

**核心知识领域**：图像形成与色彩空间、卷积与感受野、分类骨干（ResNet/ViT/ConvNeXt）、检测与分割、度量学习与检索、生成与视觉大模型、视频与 3D。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| CV-1 图像分类 | 2–3 | 卷积、池化、BN、残差、数据增广、迁移学习 | 会读骨干结构；会做 freeze/unfreeze；会用 AMP；能画 loss/acc 曲线判断过拟合 | CIFAR-10 复现 ResNet18；商品类目分类 | 复现论文指标 ±1pt；能解释每个增强项的作用 |
| CV-2 检测与分割 | 3–4 | 锚框、IoU、NMS、one/two-stage、FPN、语义/实例分割、关键点 | 会用 torchvision 预训练模型推理与微调；会算 mAP；会标注小数据集 | 商品主体检测（解决背景干扰）；模特区域裁剪 | 自建 200 图标注集 mAP@0.5 ≥ 0.5；能分析漏检类型 |
| CV-3 度量学习与检索 | 2–3 | embedding、triplet/ArcFace、PK sampler、难例挖掘、Recall@K、ANN | 会设计正负样本；会做 badcase 归因；会建 Faiss 索引 | 同款商品检索（本项目核心） | Recall@10 相对 baseline 提升 ≥ 10pt；有 recall–latency 曲线 |
| CV-4 视觉大模型与生成 | 3–4 | ViT、CLIP、SAM、Diffusion、VAE/GAN | 会做 zero-shot 分类；会用 CLIP 做图文检索；会评估生成质量 | 背景归一化、超分、长尾样本生成 | CLIP 与自建模型对比有结论；生成增广带来下游增益 |
| CV-5 视频与 3D（选） | 3–4 | 时序建模、光流、动作识别、NeRF/3DGS | 会处理视频抽帧与时间维度 | 商品短视频内容理解 | 能复现一个视频分类 baseline |

**分支特点**：视觉的"数据标注成本"最高、评价最直观（看图就能判断），适合作为入门分支建立正反馈。
**与其他分支的接口**：CLIP 是 CV 与 NLP 的第一接口；检测框是后续 VLM 与文档理解的输入。

### 6.4 分支二：自然语言处理（NLP / LLM）

**核心知识领域**：分词与词表、词表示、序列建模、注意力与 Transformer、预训练范式、指令微调与对齐、检索增强、评测。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| NLP-1 文本基础 | 2 | 分词（BPE/WordPiece）、词向量、文本分类、序列标注、评价指标 | 会构建词表；会处理变长序列与 padding；会算 PPL/F1 | 商品标题类目分类、属性抽取 | 在自有数据上超过朴素 baseline（TF-IDF + LR） |
| NLP-2 序列建模 | 2 | RNN/LSTM/GRU、text-CNN、seq2seq、注意力起源 | 理解隐状态与梯度消失；能手写 seq2seq 训练循环 | 标题翻译（中→英小语料） | BLEU 可复现；能解释 teacher forcing 与推理的差异 |
| NLP-3 Transformer 与预训练 | 3–4 | self-attention、多头、位置编码、mask、BERT vs GPT、MLM/CLM | 能不看源码手写 attention；会做领域内继续预训练与微调 | 标题领域适应预训练 + 下游微调 | 手写 attention 单测通过；微调后优于 from-scratch |
| NLP-4 LLM 应用与对齐 | 4–6 | Scaling Law、SFT、LoRA/QLoRA、RLHF/DPO、prompting、RAG、tool use、agent | 会切分文档与建向量库；会设计评测集；会识别幻觉 | 商品知识 RAG、卖点生成、结构化抽取 | 50 题自建评测集；幻觉率与人工通过率有统计 |
| NLP-5 检索与评测工程（选） | 2–3 | 向量库、rerank、混合检索、LLM-as-judge | 会做召回与重排两段式；会设计离线评测流水线 | 标题/详情页混合检索 | 检索与生成两段分别有指标 |

**分支特点**：NLP 的迭代成本最低（文本数据小、训练快），但**评测最难**——"看起来对"和"真的对"差距最大，必须从 NLP-1 就建立评测习惯。
**与其他分支的接口**：Transformer 是全领域共享的结构；RAG 的向量检索与 CV 的以图搜图是同一套 ANN 技术。

### 6.5 分支三：语音与音频

**核心知识领域**：声学基础（采样率、频谱、MFCC）、ASR（CTC / attention-based）、TTS（声学模型 + 声码器）、声纹与音频事件。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| AU-1 音频基础 | 1–2 | 采样、STFT、梅尔频谱、数据增广（加噪、变速） | 会做音频特征提取与可视化 | 商品短视频语音转文字（调用现成模型） | 能解释频谱图与波形的关系 |
| AU-2 语音识别 | 2–3 | CTC、编码器-解码器、语言模型融合 | 会微调开源 ASR；会算 WER/CER | 客服录音质检（关键词命中） | 小测试集 WER 有基线对照 |
| AU-3 语音合成 | 2–3 | 声学模型、声码器、韵律控制 | 会用开源 TTS 做音色迁移 | 商品讲解音频生成 | 主观 MOS 粗评 + 可懂度检查 |

**分支特点**：入门门槛在"数据处理"（波形对齐、时长）而非模型；多数场景可直接用开源模型 + 微调，不必从头训练。
**与其他分支的接口**：ASR 输出是 NLP 的输入；音频事件检测与 CV 的视频理解可组合成"短视频理解"。

### 6.6 分支四：多模态与跨模态

**核心知识领域**：对比学习（CLIP 系）、跨模态对齐与融合、视觉问答、文档理解（OCR + VLM）、音视频联合建模。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| MM-1 图文对齐 | 2–3 | 对比学习、InfoNCE、双塔结构、温度系数 | 会用 CLIP 做 zero-shot 与图文检索；会算 Recall@K | 以文搜商品、图文同款校验 | zero-shot 与有监督的差距有量化结论 |
| MM-2 视觉语言模型 | 3–4 | 视觉编码器 + 投影层 + LLM、指令微调、幻觉 | 会用开源 VLM 做 VQA 与描述生成；会构造指令数据 | 商品图自动生成标题与卖点 | 自建 100 条评测，人工通过率 ≥ 70% |
| MM-3 文档与结构化理解 | 2–3 | OCR、版面分析、表格抽取、多页推理 | 会组合 OCR + VLM 做字段抽取 | 商品资质/参数表信息抽取 | 字段级准确率与召回率 |
| MM-4 音视频联合（选） | 2–3 | 视频时序、字幕对齐、跨模态检索 | 会做视频片段检索 | 短视频同款片段检索 | 片段级 Recall@K |

**分支特点**：多模态是"能力放大器"——单独看不解决新问题，但能把两个分支的数据互相标注、互相增强。
**与其他分支的接口**：需要先有 CV-3（视觉 embedding）与 NLP-3（文本编码器）的基础，否则只能调 API 而无法改结构。

### 6.7 分支五：生成模型（横切底座）

**核心知识领域**：显式似然与隐式分布、VAE、GAN、Diffusion、流模型、条件生成、采样与评估。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| GEN-1 隐变量模型 | 1–2 | 自编码器、VAE、KL 正则、重参数化 | 会平衡重构损失与 KL；会看潜空间分布 | MNIST/CIFAR 上的 VAE 与插值 | 能生成连续过渡样本；能解释 KL 塌缩 |
| GEN-2 对抗生成 | 2–3 | GAN 目标、模式崩溃、谱归一化、条件 GAN | 会诊断训练不稳定；会做条件控制 | DCGAN 生成商品图；类别条件生成 | 生成样本可辨类目；训练曲线可解释 |
| GEN-3 扩散模型 | 3–4 | 前向加噪/反向去噪、噪声预测、采样器、CFG | 会用开源 diffusion 微调；理解步数与质量/速度权衡 | 白底图生成、背景替换 | 生成质量粗评（FID 或人工）+ 下游任务增益 |
| GEN-4 评估与应用边界 | 1–2 | FID/IS/CLIPScore、版权与合规 | 会做生成数据增广的对照实验 | 长尾类目增广实验 | 增广组 vs 对照组的指标差异显著 |

**分支特点**：生成模型的"训练稳定性"远比"结构创新"重要，建议始终保留一个可复现的小数据集实验。
**与其他分支的接口**：VAE 的潜空间是度量学习的替代方案；Diffusion 是 CV 与多模态生成（文生图）的共同底座。

### 6.8 分支六：图学习与结构化数据

**核心知识领域**：图表示、消息传递、GCN/GAT/GraphSAGE、异构图、链接预测、推荐系统、知识图谱。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| GNN-1 图基础 | 1–2 | 邻接矩阵、拉普拉斯、度归一化、过平滑 | 会构造图数据；会写消息传递 | Cora 节点分类 | 相对 MLP baseline 有明确提升 |
| GNN-2 图神经网络 | 2–3 | GCN、GAT、GraphSAGE、采样训练 | 会做多头注意力聚合；会处理大图采样 | 商品关系图上的类目传播 | 与 GCN 对比有消融结论 |
| GNN-3 推荐与链接预测 | 3–4 | 双塔召回、协同过滤、图召回、排序 | 会做召回-排序两段式；会算 HitRate/NDCG | 基于关系的商品召回增强 | 召回指标提升且线上延迟可控 |

**分支特点**：图学习的价值高度依赖"图建得对不对"，数据建模占七成工作量。
**与其他分支的接口**：图上的节点特征通常来自 CV/NLP 的 embedding，因此图学习一般排在两者之后。

### 6.9 分支七：强化学习与决策

**核心知识领域**：MDP、值函数与策略、策略梯度、Actor-Critic、PPO、离线 RL、RLHF。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| RL-1 基础 | 1–2 | 状态/动作/奖励、回报、贝尔曼方程、探索与利用 | 会写环境交互循环；会画回报曲线 | CartPole 上的 REINFORCE | 平均回报稳定超过随机策略 |
| RL-2 策略优化 | 2–3 | 基线、优势函数、Actor-Critic、GAE | 会做方差缩减；会调超参 | CartPole/LunarLander 上的 A2C | 相同步数下回报方差显著下降 |
| RL-3 进阶与对齐 | 3–4 | PPO、离线 RL、偏好建模、DPO | 理解 RLHF 的三段式；会做偏好数据构造 | 用偏好数据做小模型对齐实验 | 偏好胜率有提升；能说明与 SFT 的差异 |

**分支特点**：RL 的样本效率低、调参敏感，**不适合作为第一个分支**；但它是理解 LLM 对齐（RLHF/DPO）的必要背景。
**与其他分支的接口**：与 NLP-4 强耦合（对齐阶段）；与推荐系统耦合（排序策略）。

### 6.10 分支八：AI 系统与工程（横切）

**核心知识领域**：训练工程、推理优化、服务化、评测与数据飞轮。

| 阶段 | 周次 | 核心知识 | 关键概念与技能 | 实践项目 | 成果检验 |
|---|---|---|---|---|---|
| SYS-1 训练工程 | 2–3 | AMP、梯度累积、DDP/FSDP/TP、checkpoint、数据管道 | 会把单卡脚本改成多卡；会估算显存 | 4 卡训练骨干网络 | 吞吐提升有数据；显存占用可预测 |
| SYS-2 推理优化 | 2–3 | 量化（int8/int4）、蒸馏、剪枝、ONNX/TensorRT、KV Cache、批处理 | 会做精度-延迟权衡；会 profile 热点 | 同款检索服务压测 | recall–latency 曲线；P99 达标 |
| SYS-3 服务与 MLOps | 2–3 | 服务框架、模型注册、灰度、监控、AB 实验 | 会设计上线 checklist；会做回滚方案 | 端到端检索服务 | 可演示 + 有监控指标 + 有回滚预案 |
| SYS-4 评测与数据飞轮 | 2 | 离线评测集、线上指标、badcase 回流、标注闭环 | 会设计评测集切分；会做误差归因 | 搭建 badcase 回流流程 | 每周能产出"改哪里"的结论 |

**分支特点**：工程能力决定"模型能不能用"，但很难单独学——必须挂在具体任务上练。
**与其他分支的接口**：所有分支的最后一公里。

### 6.11 分支间的交叉与衔接

**六个关键交叉点**：

| 交叉点 | 涉及分支 | 说明 | 学习建议 |
|---|---|---|---|
| Transformer | NLP → CV → 多模态 → 语音 | 一个结构支撑半个领域 | 在 NLP-3 学透，之后 CV-4 / MM-2 / AU-2 都是复用 |
| Embedding 与度量学习 | CV-3 ↔ NLP-5 ↔ GNN-3 | 图像检索、向量库检索、图召回本质同源 | 先学 CV-3，再把同一套思路迁移到文本 |
| 自监督预训练 | CV-1 ↔ NLP-3 | 都是"用数据自身构造监督" | 对比 MAE 与 MLM 的设计差异 |
| 生成模型 | CV-4 ↔ GEN ↔ NLP-4 | VAE/GAN/Diffusion 与 LLM 都是分布建模 | 先 GEN-1/2，再看 LLM 的生成范式 |
| 评测方法 | 全部分支 | 指标设计、切分防泄漏、人工评估 | L1 就建立习惯，不要等到项目后期 |
| 分布式与推理优化 | SYS ↔ 全部分支 | 大模型训练与部署的公共基础设施 | 有具体模型后再学，避免空转 |

**四条典型路径（按目标选一条）**：

| 路径 | 顺序 | 适合人群 | 预计周期 |
|---|---|---|---|
| A. 视觉 → 多模态检索（本项目主线） | L0–L2 → CV-1 → CV-2 → CV-3 → GEN-1/2 → MM-1 → SYS-1/2 | 做图像类业务、要快速出成果 | 6–8 个月 |
| B. NLP → LLM 应用 | L0–L2 → NLP-1 → NLP-2 → NLP-3 → NLP-4 → MM-2 → SYS-1/2 | 做文本/知识类业务 | 6–8 个月 |
| C. 工程 → 系统专精 | L0–L2 → SYS-1 → SYS-2 → SYS-3 → 选一个分支深入（CV-3 或 NLP-4） | 偏基础设施、SRE、平台方向 | 8–10 个月 |
| D. 研究 → 交叉创新 | L0–L2 → 一个分支到 CV-4/NLP-3 → 另一个分支到 MM-2 → RL-3 或 GEN-3 | 想读论文、做新方法 | 12 个月以上 |

**衔接检查点**：每完成一个分支的阶段，问三个问题——(1) 这个分支的核心评价指标是什么；(2) 它的数据从哪来、标注成本多高；(3) 它的输出能作为哪个分支的输入。答不上来就说明只学了模型，没学体系。

### 6.12 三档时间排期

| 档位 | 每周投入 | 总周期 | 覆盖范围 | 适用 |
|---|---|---|---|---|
| 冲刺档 | 10–12h | 12 周 | L0–L2 精简 + 主分支 1 个（到能交付） | 转岗、项目急需 |
| 标准档 | 8–10h | 24 周 | L0–L2 完整 + 主分支（4 阶段）+ 辅分支（2 阶段）+ SYS 基础 | 大多数在职学习者 |
| 系统档 | 6–8h | 48 周 | 全部五层 + 两个分支到中高级 + 一个交叉方向 | 长期体系建设 |

**排期原则**：先保 L0–L2 的完整性（这两层偷工减料，后面每个分支都要返工），再保主分支的深度，最后才扩展分支数量。

### 6.13 成果检验统一标准

所有分支共用同一套四级验收，避免"跑通了就算学会"：

| 等级 | 判定标准 | 检验方式 |
|---|---|---|
| R 复现 | 能在自己的环境跑出与参考实现接近的结果 | 指标差异 ≤ 5%；能复现两次 |
| E 解释 | 能说清每个模块为什么这样设计、换掉会怎样 | 向他人讲 15 分钟并回答 5 个追问 |
| T 迁移 | 能把它改到新数据/新任务上，指标不塌 | 换数据集后仍优于朴素 baseline |
| D 交付 | 能变成有延迟/成本约束的服务，并有监控 | 有压测报告、上线 checklist、回滚方案 |

**每个分支的 capstone（结项项目）建议**：

| 分支 | Capstone | 通过线 |
|---|---|---|
| CV | 同款商品检索服务 | Recall@10 ≥ 0.85，P99 ≤ 100ms |
| NLP | 商品知识 RAG 问答 | 50 题人工通过率 ≥ 70%，幻觉可统计 |
| 语音 | 客服录音关键词质检 | 关键词召回 ≥ 0.9，误报可接受 |
| 多模态 | 以文搜商品 | 跨模态 Recall@10 有明确基线对照 |
| 生成 | 长尾类目增广流水线 | 增广组指标显著优于对照组 |
| 图 | 关系图召回增强 | 召回指标提升且延迟可控 |
| RL | 排序策略离线评估 | 离线指标优于规则策略 |
| 系统 | 端到端上线 | 有延迟、成本、可用性三项指标 |

### 6.14 与 examples 的映射

把本章的学习阶段直接落到仓库示例上（详见 §四）：

| 本章阶段 | 对应仓库示例 | 可直接改造的方向 |
|---|---|---|
| L0–L1 | `regression/`、`mnist/` | 建立训练闭环与指标习惯 |
| L2 | `mnist/`、`imagenet/main.py` | AMP、checkpoint、top-k 评测 |
| CV-1 | `imagenet/`、`mnist/` | 骨干微调与特征提取 |
| CV-2 | 无直接示例 | 用 `torchvision` 预训练检测模型自建 |
| CV-3 | `siamese_network/` | 双塔改 embedding + 余弦 |
| CV-4 | `super_resolution/`、`fast_neural_style/` | 图像到图像任务范式 |
| GEN-1/2 | `vae/`、`dcgan/` | 增广与生成质量评估 |
| NLP-2 | `time_sequence_prediction/`、`mnist_rnn/` | 序列建模手感 |
| NLP-3 | `word_language_model/` | 手写 attention、领域内预训练 |
| NLP-2/3 | `language_translation/` | encoder-decoder 与掩码 |
| NLP-1 | `legacy/snli/` | 文本分类与词向量 |
| MM-1/2 | 无直接示例 | 用 CLIP/VLM 自建 |
| GNN-1/2 | `gcn/`、`gat/` | 商品关系图迁移 |
| RL-1/2 | `reinforcement_learning/` | 加 baseline、换环境 |
| SYS-1 | `distributed/ddp/`、`distributed/ddp-tutorial-series/`、`distributed/FSDP2/` | 单卡改多卡 |
| SYS-1（大模型） | `distributed/minGPT-ddp/`、`distributed/tensor_parallelism/` | GPT/TP 实践 |
| SYS-2 | `fx/`、`fx/native_interpreter/`、`cpp/transfer-learning/` | 图优化与部署 |

**缺口提示**：检测/分割、CLIP、Diffusion、ASR/TTS、RAG 在仓库中**没有**对应示例，需要另行搭建；不要因为仓库没有就跳过这些能力。

### 6.15 常见误区

1. **跳过 L1 直接学 L3**：不知道偏差-方差、不知道指标怎么设计，后面每个项目都只能"调包看结果"。
2. **把分支当并列选项，同时开三个**：每个分支学到一半，哪个都不足以解决问题。标准做法是"1 主 + 1 辅"。
3. **只学模型不学数据**：CV 的标注成本、NLP 的评测难度、图学习的 schema 设计，才是决定成败的部分。
4. **用"跑通 SOTA 仓库"代替"能解释"**：R 级只到复现，E 级才开始形成能力。
5. **先学部署再学模型**：没有可优化的模型，工程技巧无处施展；但反过来，模型做完不学部署，成果无法落地。
6. **评测集泄漏**：同一实体同时出现在训练与测试（同 SPU、同用户、同文档），指标虚高，是最常见也最致命的问题。

---

## 七、附录：环境与常见坑

### 环境准备

| 依赖 | 涉及的示例 | 说明 |
|---|---|---|
| PyTorch ≥ 2.x | 全部 | FSDP2 需要 ≥ 2.5 |
| torchvision | 视觉类 | 版本需与 PyTorch 匹配 |
| torchtext + spacy | `legacy/snli`、`language_translation` | 需下载语言包；torchtext 接口变更频繁，`legacy` 目录即因此得名 |
| gym | `reinforcement_learning` | CartPole 环境 |
| OpenCV | `cpp/custom-dataset`、`cpp/transfer-learning` | 参考 `cpp/tools/InstallingOpenCV.md` |
| MPI | `cpp/distributed` | 需 OpenMPI/MPICH，且 LibTorch 要自行编译 |
| Sphinx | `docs/` | 修改 `docs/source/index.rst` 后需 `make html` |
| 多卡 GPU | 阶段 G 全部 | 无多卡时可用 CPU + gloo 后端跑通逻辑，但速度极慢 |

### 常见坑

1. **`legacy/snli` 跑不通**：优先怀疑 torchtext 版本，而非代码逻辑。
2. **`cpp` 示例链接失败**：`cmake` 的 `CMAKE_PREFIX_PATH` 必须指向解压后的 LibTorch，且 C++ ABI 要与 OpenCV 一致（用 `libtorch-cxx11-abi-shared-with-deps`）。
3. **`distributed/*` 起不来**：90% 是端口占用或 `MASTER_ADDR` 不通；先用单机 `--nnodes=1` 验证。
4. **`fx/native_interpreter` 编译失败**：检查 `torch.utils.cmake_prefix_path` 是否输出正确路径。
5. **`imagenet` 跑不动**：先用 `--dummy` 验证链路，再考虑真实数据；ImageNet 需要自行下载并按 `extract_ILSVRC.sh` 整理目录。
6. **`dcgan` 生成噪声**：先确认 `--dry-run` 通过，再检查 `nc`（通道数）与 `imageSize` 是否匹配数据集。

---

## 附：全量清单（按目录）

- 顶层（20）：`regression`、`mnist`、`mnist_rnn`、`mnist_hogwild`、`mnist_forward_forward`、`dcgan`、`vae`、`super_resolution`、`fast_neural_style`、`siamese_network`、`time_sequence_prediction`、`reinforcement_learning`、`imagenet`、`word_language_model`、`language_translation`、`gcn`、`gat`、`fx`、`fx/native_interpreter`、`legacy/snli`
- C++（7）：`cpp/regression`、`cpp/mnist`、`cpp/autograd`、`cpp/dcgan`、`cpp/custom-dataset`、`cpp/transfer-learning`、`cpp/distributed`
- 分布式（12）：`distributed/ddp`、`distributed/ddp-tutorial-series`、`distributed/FSDP`、`distributed/FSDP2`、`distributed/minGPT-ddp`、`distributed/tensor_parallelism`、`distributed/rpc/{parameter_server,pipeline,rnn,rl,batch,ddp_rpc}`
