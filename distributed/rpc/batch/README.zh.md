# 异步 RPC 用户函数示例

该文件夹包含两个 [`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 的示例：

1. 同步批量更新参数服务器：使用 `@rpc.functions.async_execution`
   进行参数更新与获取。这是一个简单的批量 RPC 入门示例。
   ```
   pip install -r requirements.txt
   python parameter_server.py
   ```
2. 带批处理代理的多观察者：使用 `@rpc.functions.async_execution`
   将多个观察到的状态送入策略以获取动作。
   ```
   pip install -r requirements.txt
   python reinforce.py
   ```
