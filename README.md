# LCS-Model: Link Prediction Based on Local Clustering and Subgraphs in Biological Networks

**LCS**（Local Clustering and Subgraphs）是一个专为**生物网络（Biological Networks）**设计的 GNN-based 链路预测模型。

论文提出了一种结合 **Heat Kernel (HK) 扩散 + Chopper 剪枝** 的动态局部子图提取机制，同时引入多样性正则化（GKR）和结构-属性联合嵌入，有效应对生物网络的异质性、动态性、非对称性和层次模块化等问题。

**论文信息**  
- 标题：Link Prediction Based on Subgraph Learning in Biological Networks  
- 作者：Xiaolong Liu, Jianxia Chen 等  
- 日期：March 13, 2026  
- 实现仓库：https://github.com/XL0104/LCS-Model.git

## 主要特性

- **高效子图提取**：采用 Heat Kernel 扩散 + Chopper 线性时间剪枝，避免传统 k-hop 子图爆炸问题。
- **多样性正则化**：通过 GKR 机制提升模型泛化能力，缓解异质关系类型过多导致的过拟合。
- **结构-属性联合嵌入**：融合 DRNL 局部结构标签、Node2Vec 全局结构特征与节点属性。
- **动态图卷积预测**：基于 EdgeConv + GRU 的动态预测模块，适用于具有时序或动态特性的生物网络。
- **支持多种生物数据集**：MDA、LuoDTI、ZhangMDA、ZhangDDA 以及自定义数据集。

本仓库基于 **ScaLed**（SEAL 的改进版）实现，支持随机游走采样（m, M 参数）和传统 k-hop 模式，兼容生物网络实验。

## 支持的数据集

### 1. 论文中使用的生物网络数据集
- **MDA** (Microbe-Disease Associations) — 微生物-疾病关联
- **LuoDTI** (Drug-Target Interactions) — 药物-靶点交互
- **ZhangMDA** (miRNA-Disease Associations) — miRNA-疾病关联
- **ZhangDDA** (Drug-Disease Associations) — 药物-疾病关联

### 2. 本仓库原生支持的数据集
- Planetoid：`Cora`, `Pubmed`, `Citeseer`
- SEAL 经典数据集：`USAir`, `NS`, `Power`, `Celegans`, `Yeast` 等
- **自定义数据集**（强烈推荐用于生物网络）：
  - `Custom`
  - `MiRNA`
  - `KIBA`
  - `Microbe`（对应 MDA 类数据集）

**自定义数据集准备**：  
在 `edges/` 目录下放置：
- `xxx_edges.txt`（每行两个节点 ID，无权重）
- `xxx_features.txt`（节点特征矩阵，可选，配合 `--use_feature` 使用）

## 环境安装

```bash
# 如果有 quick_install.sh（推荐）
source quick_install.sh

# 否则手动安装核心依赖（PyTorch + PyG）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install networkx scipy numpy tqdm scikit-learn ogb
