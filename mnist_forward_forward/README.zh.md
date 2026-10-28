# 基础 Forward-Forward 示例

本示例实现了 Geoffrey Hinton 的论文 [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)。

该论文旨在为神经网络引入一种新的学习过程。它用两次前向传播取代了反向传播的前向传播和反向传播。

```bash
pip install -r requirements.txt
python main.py
```

main.py 脚本接受以下参数：

```bash
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 1000)
  --lr LR               learning rate (default: 0.03)
  --no_accel            disables accelerator
  --seed SEED           random seed (default: 1)
  --save_model          For saving the current Model
  --train_size TRAIN_SIZE
                        size of training set
  --threshold THRESHOLD
                        threshold for training
  --test_size TEST_SIZE
                        size of test set
  --save-model          For Saving the current Model
  --log-interval LOG_INTERVAL
                        logging training status interval
```
