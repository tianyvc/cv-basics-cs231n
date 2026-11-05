# 作业 1：图像分类与全连接网络

[英文原文](assignment1.md) · [逐步学习实践指南](../docs/CS231n_2025_逐步学习实践指南.md)

> 本文对应 2025 年官方作业说明。官方截止时间与 Colab、Gradescope 提交流程仅作历史参考。本仓库的本地实验请按[本地实验与存储管理](../docs/本地实验与存储管理.md)操作，不需要挂载 Google Drive。

官方截止时间为 **2025 年 4 月 23 日（星期三），美国太平洋时间晚上 11:59**。

仓库已保留[官方作业原包](assignment1_colab.zip)和[本地适配后的作业目录](assignment1/assignment1/knn.ipynb)。不要重新解压原包覆盖已经填写的代码和答案。

## 目录

- [环境准备](#setup)
- [学习目标](#goals)
- [第 1 题：k 近邻分类器](#q1-k-nearest-neighbor-classifier)
- [第 2 题：实现 Softmax 分类器](#q2-implement-a-softmax-classifier)
- [第 3 题：两层神经网络](#q3-two-layer-neural-network)
- [第 4 题：更高层次的表示——图像特征](#q4-higher-level-representations-image-features)
- [第 5 题：训练全连接网络](#q5-training-a-fully-connected-network)
- [提交作业](#submitting-your-work)

<a id="setup"></a>

## 环境准备

官方建议先观看 [Colab 工作流程演示](https://www.youtube.com/watch?v=DsGd2e9JNH4)。本地学习可以跳过其中云盘挂载和虚拟机操作部分。

请经常通过 `File → Save` 保存 Notebook。官方提示这样可以防止 Colab 虚拟机断开连接时丢失进度；在本地实验中也应养成定期保存的习惯。

完成除 `collect_submission.ipynb` 以外的全部练习 Notebook 后，再根据需要整理结果。

本地运行时，激活已有的 `cs231n` 环境，选择对应的 Jupyter 内核。数据只需要共享的 CIFAR-10；在仓库根目录按需执行：

```bash
python tools/local_resources.py fetch cifar10
```

CIFAR-10 在后续多个实验中复用，不必为每道题重复下载。具体使用周期和清理方式见[存储管理指南](../docs/本地实验与存储管理.md)。

<a id="goals"></a>

## 学习目标

本次作业要求你搭建一个简单的图像分类流程，理解 k 近邻及 SVM/Softmax 等分类方法。实际题目要求以本目录的五本练习 Notebook 为准，2025 年模板没有单独的 SVM 实现题。

- 理解基本的**图像分类流程**和数据驱动方法，即训练与预测两个阶段。
- 理解训练集、验证集、测试集的划分，以及如何利用验证集进行**超参数调优**。
- 熟练编写高效的 NumPy **向量化代码**。
- 实现并使用 **k 近邻（kNN）分类器**。
- 实现并使用 **Softmax 分类器**。
- 实现并使用**两层神经网络分类器**。
- 实现并使用**全连接网络分类器**。
- 理解这些分类器的差别，以及各自的取舍。
- 初步理解用**更高层次的特征表示**替代原始像素为何能改善性能，例如颜色直方图和方向梯度直方图（HOG）。

<a id="q1-k-nearest-neighbor-classifier"></a>

## 第 1 题：k 近邻分类器

按照 [knn.ipynb](assignment1/assignment1/knn.ipynb) 中的说明实现 kNN 分类器。

<a id="q2-implement-a-softmax-classifier"></a>

## 第 2 题：实现 Softmax 分类器

按照 [softmax.ipynb](assignment1/assignment1/softmax.ipynb) 中的说明实现 Softmax 分类器。

<a id="q3-two-layer-neural-network"></a>

## 第 3 题：两层神经网络

按照 [two_layer_net.ipynb](assignment1/assignment1/two_layer_net.ipynb) 中的说明实现两层神经网络分类器。

<a id="q4-higher-level-representations-image-features"></a>

## 第 4 题：更高层次的表示——图像特征

在 [features.ipynb](assignment1/assignment1/features.ipynb) 中比较原始像素与更高层次特征表示，观察特征表示带来的分类性能改善。

<a id="q5-training-a-fully-connected-network"></a>

## 第 5 题：训练全连接网络

按照 [FullyConnectedNets.ipynb](assignment1/assignment1/FullyConnectedNets.ipynb) 中的说明实现全连接网络。

<a id="submitting-your-work"></a>

## 提交作业

**官方要求：提交的 Notebook 应当已经运行，并保留可见的单元格输出。** 填完所有必需代码后，官方历史流程如下：

1. 在 Colab 中打开并执行 `collect_submission.ipynb`。官方原版收集脚本会生成含 `.py` 和 `.ipynb` 的 `a1_code_submission.zip`，并把 Notebook 汇总成 `a1_inline_submission.pdf`。
2. 当看到提示 `### Done! Please submit a1_code_submission.zip and a1_inline_submission.pdf to Gradescope. ###`，下载这两个文件并提交至 [Gradescope](https://www.gradescope.com/courses/1012166)。

**本地自学说明：** 本仓库的收集工具已改为显式导出源码 ZIP 和可选 HTML，默认不导出，也不会自动安装 TeX 或生成官方 PDF。自学不需要访问 Gradescope；如果需要正式提交，应按当期课程的实际要求另行准备。
