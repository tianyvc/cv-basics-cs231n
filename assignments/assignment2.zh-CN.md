# 作业 2：归一化、正则化、卷积网络与循环网络

[英文原文](assignment2.md) · [逐步学习实践指南](../docs/CS231n_2025_逐步学习实践指南.md)

> 本文对应 2025 年官方作业说明。截止时间与云端提交流程是历史信息。本地实验请遵循[本地实验与存储管理](../docs/本地实验与存储管理.md)，无需 Google Drive。

官方截止时间为 **2025 年 5 月 7 日（星期三），美国太平洋时间晚上 11:59**。原文标作 PST；此处保留其官方时间信息，不作为当前提交提醒。

可对照[官方原包](assignment2_colab.zip)。实际运行请使用已经解压并适配本地环境的 Notebook，避免覆盖自己的作答。

## 目录

- [环境准备](#setup)
- [学习目标](#goals)
- [第 1 题：批量归一化](#q1-batch-normalization)
- [第 2 题：Dropout](#q2-dropout)
- [第 3 题：卷积神经网络](#q3-convolutional-neural-networks)
- [第 4 题：使用 PyTorch 处理 CIFAR-10](#q4-pytorch-on-cifar-10)
- [第 5 题：使用基本 RNN 生成图像描述](#q5-image-captioning-with-vanilla-rnns)
- [提交作业](#submitting-your-work)

<a id="setup"></a>

## 环境准备

官方建议开始前熟悉 [Colab 工作流程](https://cs231n.github.io/setup-instructions/#working-remotely-on-google-colaboratory)，并观看 [Colab 演示视频](https://www.youtube.com/watch?v=DsGd2e9JNH4)。原文说明官方不提供本地开发支持，同时提供 `requirements.txt` 供创建虚拟环境参考。

本仓库已完成本地适配，依赖安装以本地指南为准。请定期通过 `File → Save` 保存 Notebook；完成除 `collect_submission.ipynb` 之外的所有练习后，再整理导出结果。

第 1–4 题复用已有 CIFAR-10。进入第 4 题前准备与本机 CPU/GPU 相匹配的 PyTorch 和 torchvision。第 5 题才需要课程整理过的 COCO 图像描述和图像特征数据，而非完整 COCO 原始图像集；按需运行：

```bash
python tools/local_resources.py fetch coco
```

COCO 数据还会在作业 3 的 Transformer 图像描述题中复用，完成后再清理。

<a id="goals"></a>

## 学习目标

本次作业将练习编写反向传播代码，训练神经网络与卷积神经网络，并将循环网络用于图像描述。

- 实现用于训练深层网络的**批量归一化（Batch Normalization）**和**层归一化（Layer Normalization）**。
- 实现 **Dropout**，对网络进行正则化。
- 理解**卷积神经网络**的结构，并实际训练它们。
- 掌握主流深度学习框架 **PyTorch** 的基本使用方式。
- 理解并实现 **RNN**，将它与 CNN 结合用于图像描述。

<a id="q1-batch-normalization"></a>

## 第 1 题：批量归一化

在 [BatchNormalization.ipynb](assignment2/assignment2/BatchNormalization.ipynb) 中实现批量归一化，并用它训练深层全连接网络。

<a id="q2-dropout"></a>

## 第 2 题：Dropout

在 [Dropout.ipynb](assignment2/assignment2/Dropout.ipynb) 中实现 Dropout，考察它对模型泛化能力的影响。

<a id="q3-convolutional-neural-networks"></a>

## 第 3 题：卷积神经网络

在 [ConvolutionalNetworks.ipynb](assignment2/assignment2/ConvolutionalNetworks.ipynb) 中实现卷积网络常用的若干新层。

<a id="q4-pytorch-on-cifar-10"></a>

## 第 4 题：使用 PyTorch 处理 CIFAR-10

这一部分使用流行且功能完善的深度学习框架 PyTorch。打开 [PyTorch.ipynb](assignment2/assignment2/PyTorch.ipynb)，先理解框架的工作方式，再自行设计卷积网络，在 CIFAR-10 上尽量获得较好的分类效果。

<a id="q5-image-captioning-with-vanilla-rnns"></a>

## 第 5 题：使用基本 RNN 生成图像描述

按照 [RNN_Captioning_pytorch.ipynb](assignment2/assignment2/RNN_Captioning_pytorch.ipynb) 中的说明实现基本循环神经网络，并将它用于 COCO 图像描述。

<a id="submitting-your-work"></a>

## 提交作业

**官方要求：Notebook 应当已经运行，并保留可见的单元格输出。** 填完所有必需代码后，官方历史流程如下：

1. 在 Colab 中执行 `collect_submission.ipynb`。官方原版脚本生成源码文件 `a2_code_submission.zip` 和汇总 PDF `a2_inline_submission.pdf`。
2. 成功提示为 `### Done! Please submit a2_code_submission.zip and a2_inline_submission.pdf to Gradescope. ###`。
3. 下载这两个文件并提交至 [Gradescope](https://www.gradescope.com/courses/1012166)。

**执行顺序：** 官方特别要求最近一次内核执行应按单元格顺序完成，否则可能影响自动评分。如果顺序混乱，在代码已经补全后重启内核，使用“Restart and Run All”依次重跑。

**本地自学说明：** 本仓库当前收集工具默认不导出；需要时可显式生成源码 ZIP 和可选 HTML，未提供官方 PDF 自动生成流程。自学无需 Gradescope；正式提交应另行核对课程要求。
