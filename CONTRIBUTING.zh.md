# 为 examples 做贡献

我们希望尽可能让为本项目做贡献变得简单、透明。

## Pull Request

我们非常欢迎你的 pull request。

如果你是新手，我们建议你查看带有 [good first issue](https://github.com/pytorch/examples/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 标签的 issue。

### 新增示例

0. 创建一个 GitHub issue 提出新示例，并确保它与现有示例有实质性区别。
1. Fork 本仓库，并从 `main` 创建你的分支。
2. 如果你添加了需要测试的代码，请在 `run_python_examples.sh` 中添加测试。
3. 创建一个 `README.md`。
4. 在 `docs/source/index.rst` 文件中添加一个卡片，简要描述你的示例并附上仓库链接，
   然后通过运行以下命令构建文档：

   ```
   cd docs
   virtualenv venv
   source venv/bin/activate
   pip install -r requirements.txt
   make html
   ```

   使用完 `virtualenv` 后，运行 `deactivate`。

5. 确认文档构建没有出现问题。你可以通过安装 [sphinx-serve](https://pypi.org/project/sphinx-serve/)，
   然后运行 `sphinx-serve -b build` 在本地检查预览。
6. 确保你的测试在本地通过。
7. 如果尚未完成，请完成贡献者许可协议（"CLA"）。
8. 及时处理代码评审中的反馈。

## 修复缺陷

1. Fork 本仓库，并从 `main` 创建你的分支。
2. 确保你有一台支持 GPU 的机器，可以是本地机器或云端机器。AWS 上的 `g4dn.4xlarge` 是一个不错的起点。
3. 进行代码修改。
4. 安装 `uv`。
5. 然后，完整运行脚本，确保 `VIRTUAL_ENV=.venv ./run_python_examples.sh` 在本地通过。
6. 如果尚未完成，请完成贡献者许可协议（"CLA"）。
7. 及时处理代码评审中的反馈。

## 贡献者许可协议（"CLA"）

为接受你的 pull request，我们需要你提交 CLA。你只需
完成一次，即可参与 Facebook 的任意开源项目。

在此完成你的 CLA：<https://code.facebook.com/cla>

## Issue

我们使用 GitHub issue 来跟踪公开的缺陷。请确保你的描述
清晰，并包含足以复现该问题的说明。

## 许可

通过向 examples 做贡献，你同意你的贡献将按照本源码树根目录中的
LICENSE 文件进行许可。
