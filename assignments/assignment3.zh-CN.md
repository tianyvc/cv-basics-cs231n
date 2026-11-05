# 作业 3：Transformer、自监督学习与扩散模型

[英文原文](assignment3.md) · [逐步学习实践指南](../docs/CS231n_2025_逐步学习实践指南.md)

> 本文对应 2025 年官方作业说明。截止时间、Colab 与 Gradescope 流程仅作历史参考。本地实验请按[本地实验与存储管理](../docs/本地实验与存储管理.md)操作。

官方截止时间为 **2025 年 5 月 30 日（星期五），美国太平洋时间晚上 11:59**。原文标作 PST，此处保留其历史信息。

仓库已保存[官方原包](assignment3_colab.zip)；运行时使用本地适配后的解压目录，不要覆盖已经填写的答案。

## 目录

- [环境准备](#setup)
- [学习目标](#goals)
- [第 1 题：使用 Transformer 生成图像描述](#q1-image-captioning-with-transformers)
- [第 2 题：用于图像分类的自监督学习](#q2-self-supervised-learning-for-image-classification)
- [第 3 题：去噪扩散概率模型](#q3-denoising-diffusion-probabilistic-models)
- [第 4 题：CLIP 和 DINO](#q4-clip-and-dino)
- [提交作业](#submitting-your-work)

<a id="setup"></a>

## 环境准备

官方建议先熟悉 [Colab 推荐工作流程](https://cs231n.github.io/setup-instructions/#working-remotely-on-google-colaboratory)，观看 [Colab 演示视频](https://www.youtube.com/watch?v=DsGd2e9JNH4)，并定期通过 `File → Save` 保存 Notebook，以免云端断开连接后丢失进度。

原文说明官方不提供本地开发支持，但附带 `requirements.txt` 供配置虚拟环境参考。本仓库已作本地适配，依赖、路径和资源获取以本地指南为准。完成除 `collect_submission.ipynb` 之外的全部练习后，再考虑导出结果。

本次作业**大部分使用 PyTorch**。第 1 题复用作业 2 的 COCO 特征与描述；第 2 题复用 CIFAR-10，并可能按 Notebook 对照实验需要获取 SimCLR 权重；第 3 题按阶段获取 emoji 数据或 DDPM 检查点；第 4 题按需获取 CLIP、DINO，只有视频部分才涉及 DAVIS 和 TensorFlow Datasets。

这些资源不应一次性全部下载。Notebook 中模型和视频资源的下载开关默认关闭；每项的大小、用途、获取命令、复用与清理时间见[按需资源指南](../docs/未下载资源与按需获取指南.md)。

<a id="goals"></a>

## 学习目标

本次作业先实现语言建模网络，并在 COCO 上进行图像描述；接着利用自监督学习，从没有人工类别标签的数据中学习视觉表示；然后实现去噪扩散概率模型（DDPM），生成图像；最后探索 CLIP 和 DINO 如何利用大量数据学习视觉表示。

- 理解并实现 **Transformer**，将它与 CNN 结合用于图像描述。
- 理解如何利用**自监督学习**改善图像分类。
- 理解并实现 **DDPM**，将其用于图像生成。
- 理解并探索 **CLIP 和 DINO** 的表示学习方法。更准确地说，CLIP 使用图像与文本配对的监督信号，DINO 通过自蒸馏学习图像表示；不应把两者的数据监督方式视为完全相同。

<a id="q1-image-captioning-with-transformers"></a>

## 第 1 题：使用 Transformer 生成图像描述

[Transformer_Captioning.ipynb](assignment3/assignment3/Transformer_Captioning.ipynb) 将引导你实现 Transformer，并在 COCO 上生成图像描述。

<a id="q2-self-supervised-learning-for-image-classification"></a>

## 第 2 题：用于图像分类的自监督学习

在 [Self_Supervised_Learning.ipynb](assignment3/assignment3/Self_Supervised_Learning.ipynb) 中学习如何通过自监督预训练，改善图像分类性能。

官方 Colab 说明要求首次打开时进入 `Runtime → Change runtime type`，将 `Hardware accelerator` 设置成 `GPU`。**本地对应操作**是安装匹配硬件的 PyTorch，检查 `torch.cuda.is_available()`，并使用代码中的设备选择；本地 Jupyter 没有 Colab 的这一菜单。CPU 可用于阅读和小规模验证，完整训练耗时需自行评估。

<a id="q3-denoising-diffusion-probabilistic-models"></a>

## 第 3 题：去噪扩散概率模型

在 [DDPM.ipynb](assignment3/assignment3/DDPM.ipynb) 中实现去噪扩散概率模型，并用于图像生成。

<a id="q4-clip-and-dino"></a>

## 第 4 题：CLIP 和 DINO

在 [CLIP_DINO.ipynb](assignment3/assignment3/CLIP_DINO.ipynb) 中实现和探索 CLIP、DINO 相关功能，理解它们学习到的视觉表示。

<a id="submitting-your-work"></a>

## 提交作业

**官方要求：提交前运行 Notebook，并保留可见的单元格输出。** 填完所有必需代码后，官方历史流程如下：

1. 在 Colab 中执行 `collect_submission.ipynb`。官方原版脚本会生成含 `.py` 和 `.ipynb` 的 `a3_code_submission.zip`，并汇总成 `a3_inline_submission.pdf`。
2. 成功提示为 `### Done! Please submit a3_code_submission.zip and a3_inline_submission.pdf to Gradescope. ###`。
3. 下载两个文件后，提交至 [Gradescope](https://www.gradescope.com/courses/1012166)。

**本地自学说明：** 本地收集工具改为显式生成源码 ZIP 和可选 HTML，默认关闭，不会自动安装 PDF 工具。自学不需要云端提交；正式提交应另行核对课程平台要求。
