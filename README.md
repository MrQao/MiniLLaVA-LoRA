# MiniLLaVA + LoRA

教学型视觉语言模型：冻结大部分 CLIP ViT-B/32 视觉编码器，将 49 个 768 维视觉 token 经线性投影接入 Qwen2.5-0.5B-Instruct。手动实现 rank=8、alpha=16 的 LoRA，注入 q/v/o/gate/up/down 六类线性层，共 144 个模块。不是官方 LLaVA 项目。

## 文件与运行

- `miniLLaVa.py`：模型、CIFAR-10 分类式问答、手动 LoRA 以及训练/测试入口。
- `testLLM.py`：保留原文本 embedding 探索草稿，不属于推荐运行入口。
- `data/LoRA02.pth`：已有完整检查点（约 1.3 GB），含视觉层、投影层及 LoRA。
- `lora_weights.pt`：旧 LoRA-only 文件，单独加载不足以恢复视觉投影层；不作为默认测试权重。

```bash
python miniLLaVa.py --help
python miniLLaVa.py test --eval-size 100
# 基于完整旧检查点继续训练；覆盖 lora_weights.pt 和 data/finetuned.pth
python miniLLaVa.py train --epochs 1
python miniLLaVa.py test --checkpoint data/finetuned.pth --eval-size 100
```

首次需要下载 `openai/clip-vit-base-patch32` 和 `Qwen/Qwen2.5-0.5B-Instruct`，使用 Hugging Face 默认缓存；本地现有全局缓存仍可使用，不依赖原工程路径。默认 MPS/CPU 自动选择，模型统一 float32。MPS 的历史运行存在兼容问题，本次只做静态检查，尚未验证设备运行。

## 拆分时的入口修正与局限

新副本将初始化和运行置于 main 保护内，删除无关的 mistral tokenizer 导入；默认严格加载完整检查点；移除训练函数中错误冻结 LoRA 的循环；padding 标签设为 -100；保存 LoRA 及完整微调检查点；测试使用官方 test split 前 N 张，避免原脚本从训练集取样评测。训练仍使用 train split 前 500 张，视觉编码器最后一层、投影层以及 LoRA 都参与训练。当前图像预处理沿用 resize + ToTensor，没有标准 CLIP 归一化；生成只支持单图。

旧报告指出已有 LoRA B 矩阵为零，旧适配器未产生增量，不能称其已完成有效 LoRA 训练。原 59%→62% 是单独历史脚本在固定均衡 100 张上的继续微调实验（仅训练 LoRA），统计不显著，不能作为此入口的测量结果。原始历史实验脚本未混入此独立工程。

## 环境安装

建议 Python 3.11，在项目根目录执行（所有相对路径均以项目根目录为基准）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

核心版本依据原实验环境整理；并非跨平台锁文件。首次运行可能下载数据与预训练模型，需要网络与足够磁盘空间。原环境为 Apple Silicon，其他平台需选择适配硬件的 PyTorch 安装包。本次拆分未重新训练或验证完整模型指标。

## 本地资源与 GitHub

本地保留本项目对应的数据及权重副本，不引用原 MiniCLIPTest，也不使用符号链接。`.gitignore` 排除数据、权重、运行产物、缓存和 IDE 配置；这些文件不会随普通 Git 提交上传。其他设备需要下载基础模型/数据，并另行获取自有训练检查点。可通过 GitHub Releases 或 Git LFS 管理权重；请确认数据和基础模型的许可后再分发。未擅自添加开源许可证，请在公开前自行选择代码许可证。

上传到新建空仓库时，可在本目录执行：

```bash
git init
git add .
git status --short
# 确认暂存内容后
git commit -m "Initial commit"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

数据与权重仅保留在本地，Git 仓库只包含代码、依赖与说明文件。
