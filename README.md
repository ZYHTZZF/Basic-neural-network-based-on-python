# 基于 Python 的全连接神经网络（MNIST）

一个用 **纯 Python/NumPy** 实现的“从零开始”全连接神经网络（MLP）示例，用 MNIST 手写数字集训练与测试。仓库内已提供一份训练好的权重文件，可直接跑推理；也支持从头训练。按当前权重测试，准确率约 **96%**（以你仓库描述为准）。

## 功能特性
- 从零实现前向与反向传播，便于教学和源码阅读  
- **预训练权重**（`mnist_nn_model.npz`）即开即用  
- 支持重新训练与评估  
- 提供 `test_photos/` 示例图片目录，方便做单样本推理演示

## 目录结构
Basic-neural-network-based-on-python/
├─ 1.py # 主脚本：训练 / 推理（加载 npz 权重）
├─ mnist_nn_model.npz # 预训练权重（已提供）
├─ test_photos/ # 测试图片示例（可放手写数字图片）
└─ README.md # 项目说明（本文件）


## 环境要求
- Python 3.8+（建议 3.10/3.11）
- 依赖库：
  - `numpy`
  - `matplotlib`（若需要画训练曲线/可视化）
  - （可选）`opencv-python` 或 `Pillow`（若要做图片推理）

安装依赖示例：
```bash
pip install numpy matplotlib opencv-python pillow


