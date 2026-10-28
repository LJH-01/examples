# 使用 FX 和 TorchScript 自定义类将 PyTorch 代码转换为原生运行时

在本示例中，我们将构建一个完成以下工作的流水线：

1. 将 PyTorch 模块中的代码转换（或"下沉"，lower）为另一种表示（我们将在示例内定义该表示）
2. 为该代码表示注册一个解释器，使其可在 TorchScript 或 Python 中使用
3. 将转换后的代码包装成仍可用于 TorchScript 编译的格式。

我们将在本示例中构建一个简单的解释器，但你可以想象将同样的流程扩展到更复杂的后端，例如进行代码优化或卸载到加速器的后端。

我们将使用 [TorchScript 自定义类](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)将该解释器暴露给 Python 和 TorchScript。在阅读本示例项目之前，你可能想先回顾该教程和文档。

### 定义解释器

我们在 `interpreter.cpp` 中定义解释器。该解释器非常有限：它只支持两种逐元素操作（`add` 和 `mul`），并且只支持 `Tensor` 值。当该解释器运行代码时，它会遍历指令列表，并简单地调用 C++ 中相应的 PyTorch 运算符。

要将解释器构建为可加载使用的共享对象文件，请在本示例的根目录下使用以下命令：


```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
$ make -j
```

构建完成后，你应该会看到 `build/libinterpreter.so`（根据你的操作系统，扩展名可能不同）。接下来我们会将该动态库加载到进程中用于执行。

### 定义变换

我们在 `use_interpreter.py` 中定义将 `PyTorch` 模块转换为解释器所理解格式的代码。请注意，该文件通过 `torch.classes.load_library` 调用加载上一步构建的共享对象。`use_interpreter.py` 包含驱动代码，最后可以直接运行以测试下沉变换。

### 问题、评论、反馈

请将问题和讨论发到 [PyTorch 论坛](https://discuss.pytorch.org/)。若要报告 PyTorch（包括 FX 和自定义类）的任何问题，请使用[问题跟踪器](https://github.com/pytorch/pytorch/issues)。
