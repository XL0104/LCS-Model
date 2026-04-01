# 数据集 (Datasets)

本项目（LCS-Model）在四个生物网络基准数据集上进行链路预测实验，分别是：

- **MDA**：Microbe-Disease Associations（微生物-疾病关联）
- **LuoDTI**：Drug-Target Interactions（药物-靶点交互）
- **ZhangMDA**：miRNA-Disease Associations（miRNA-疾病关联）
- **ZhangDDA**：Drug-Disease Associations（药物-疾病关联）

这些数据集来自公开领域，原始数据经过预处理后存放在 `edges/` 文件夹中（格式为 `xxx_edges.txt` 和 `xxx_features.txt`）。

## 数据集来源与下载地址

### 1. MDA (Microbe-Disease Associations)
- **主要来源**：HMDAD (Human Microbe-Disease Association Database)
- **下载地址**：http://www.cuilab.cn/hmdad （进入 Download 页面可获取完整数据集）
- **补充数据库**：Peryton（实验支持的微生物-疾病关联）
  - 下载地址：http://peryton.di.uoa.gr/

### 2. LuoDTI (Drug-Target Interactions)
- **主要来源**：DrugBank、BindingDB、CTD 等
- **推荐下载地址**：
  - DrugBank：https://go.drugbank.com/
  - BindingDB：https://www.bindingdb.org/bind/index.jsp
  - LCIdb（大型 DTI 数据集）：https://zenodo.org/records/10731712
  - BioSNAP（部分 DTI 数据）：https://biosnap.stanford.edu/data/

### 3. ZhangMDA (miRNA-Disease Associations)
- **主要来源**：HMDD v4.0 (Human microRNA Disease Database)
- **下载地址**：http://www.cuilab.cn/hmdd （Download 页面提供 TXT 和 Excel 格式完整数据集）

### 4. ZhangDDA (Drug-Disease Associations)
- **主要来源**：CTD (Comparative Toxicogenomics Database)
- **下载地址**：http://ctdbase.org/
- **补充资源**：DDA-Bench（药物-疾病关联基准数据集集合）
  - https://dda.csbios.net/ （包含多个基准数据集）

### 其他相关生物网络数据集
- **PPI (Protein-Protein Interactions)**：
  - BioGRID：https://thebiogrid.org/
  - STRING：https://string-db.org/
- **Drug features**：https://www.drugbank.ca/ 或 https://go.drugbank.com/
- **Disease MeSH descriptors**：https://www.nlm.nih.gov/mesh/meshhome.html 或 https://meshb.nlm.nih.gov/

## 数据格式说明

本项目中数据集统一处理为以下格式（存放在 `edges/` 文件夹）：

- `xxx_edges.txt`：边列表文件，每行两个节点ID（无权重），用于构建图结构。
- `xxx_features.txt`：节点特征矩阵（可选），用于 `--use_feature` 模式。

如果你需要原始数据，请从上述官方链接下载后进行预处理（节点ID映射、特征提取等），再放入 `edges/` 文件夹。

## 使用方法

运行实验时指定对应数据集名称，例如：

```bash
# Microbe（对应 MDA 类）
python seal_link_pred.py --dataset Microbe --m 3 --M 20 --use_feature --seed 1

# MiRNA（对应 ZhangMDA 类）
python seal_link_pred.py --dataset MiRNA --m 3 --M 20 --use_feature --seed 1
