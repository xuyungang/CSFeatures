# CSFeatures: Identification of cell type-specific differential features in single-cell and spatial omics data

## 介绍
CSFeatures是一个用来在单细胞和空间数据中识别差异基因或差异可及性区域的工具。

## 安装

推荐使用conda创建新的虚拟环境来运行本项目。

```
conda create -n CSFeatures python=3.10
conda activate CSFeatures
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据
要准备的数据为scRNA-seq或scATAC-seq的表达矩阵（行为细胞，列为基因）和细胞类型。如果是空间类型的数据，还需要提供空间坐标相关的数据。空间信息可以用STAGATE，SpatialPCA或SpaGCN等方法来提取。

在python中，你需要提供一个Anndata对象作为输入。Anndata是anndata设计用来存储单细胞数据的一个类。

提供的Anndata的格式应该满足以下要求:
- Anndata的形状为(细胞个数,基因个数或peak个数)。
- Anndata.obs中应有celltype项，用来提供细胞类别。
- 如果是空间scRNA-seq/scATAC-seq数据，还需要在Anndata.obsm提供spatial项，用来提供空间信息。

以下是一个scRNA-seq/scATAC-seq数据的例子:
```bash
AnnData object with n_obs × n_vars = 1000 × 2000
    obs: 'celltype'
```
以下是一个空间scRNA-seq/scATAC-seq数据的例子:

```bash
AnnData object with n_obs × n_vars = 1000 × 2000
    obs: 'celltype'
    obsm: 'spatial'
```

### 2. 寻找差异基因/peak

该部分主要使用了getMarkersEI和get_spatial_MarkersEI两个函数。

## `getMarkersEI` 函数

该函数用于处理包含在 `AnnData` 对象中的单细胞基因表达数据，基于多种统计方法识别标记基因。该函数执行以下任务：
- 降维
- 构建相似度图
- 调整表达值
- 计算 `method_list` 中指定的基因信息指标

### 参数

- **`adata` (str)**:  
  包含单细胞基因表达矩阵的 `AnnData` 对象，其 `.X` 属性应为二维数组，行对应于细胞，列对应于基因。

- **`n_comps` (int, 可选)**:  
  在 PCA 中计算的主成分数量。默认值为 50。

- **`n_neighbors` (int, 可选)**:  
  构建 k-最近邻图时使用的最近邻数量。默认值为 30。

- **`metric` (str, 可选)**:  
  用于计算细胞之间相似度的距离度量。默认值为 `'euclidean'`。

- **`method_list` (list, 可选)**:  
  计算各种基因统计量的函数列表。默认包含：

  - `calculate_mean_and_var`
  - `calculate_smoothness`
  - `calculate_V`
  - `calculate_prop`
  - `calculate_local_mean_max`
  - `calculate_EI`

  这些函数应在其他地方定义，用于计算基因表达分析的不同指标。

### 返回值

- **`info`**:  
  包含根据 `method_list` 中的方法计算的基因信息指标的数据结构（例如，`DataFrame`）。

- **`adata`**:  
  带有EI值的adata数据


## `get_spatial_MarkersEI` 函数

该函数处理包含空间信息的单细胞基因表达数据，基于多种统计方法识别空间标记基因。通过以下步骤进行数据降维、相似度矩阵构建、表达值调整，并计算 `method_list` 中指定的基因信息指标。

### 参数

- **`adata` (AnnData)**:  
  包含单细胞基因表达矩阵的 `AnnData` 对象。其 `.X` 属性应为二维数组，行对应于细胞，列对应于基因。

- **`n_comps` (int, 可选)**:  
  在 PCA 中计算的主成分数量。默认值为 50。

- **`n_neighbors` (int, 可选)**:  
  构建 k-最近邻图时使用的最近邻数量。默认值为 30。


- **`metric` (str, 可选)**:  
  用于计算细胞之间相似度的距离度量。默认值为 `'euclidean'`。

- **`spatial_key` (str, 可选)**:  
  用于指定空间信息的键。默认值为 `'spatial'`。


- **`method_list` (list, 可选)**:  
  计算各种基因统计量的函数列表。默认包含：

  - `calculate_mean_and_var`
  - `calculate_smoothness`
  - `calculate_V`
  - `calculate_prop`
  - `calculate_local_mean_max`
  - `calculate_EI`


### 返回值

- **`info`**:  
  包含根据 `method_list` 中的方法计算的基因信息指标的数据。

- **`adata`**:  
  带有EI值的adata数据


本仓库在谷歌云盘中提供了四个数据，分别为[scATAC-seq](https://drive.google.com/file/d/1mXGWKpOMR4I-mqhyAIQ_UFV6VbHwizdh/view?usp=drive_link)，[scRNA-seq](https://drive.google.com/file/d/1LWOnXLHYn8W6GFQ2NTfi84JyY9B4XGOK/view?usp=drive_link)，[空间scATAC-seq](https://drive.google.com/file/d/1w7oxnwR_Nma5uTm0yOf4I2O44tGC5Dif/view?usp=drive_link)，[空间scRNA-seq数据](https://drive.google.com/file/d/1U3_0FIBEcTLzTiAHQG00sMNSLvq7lFtl/view?usp=drive_link)。
以下以这四个数据为例分别演示使用本工具寻找差异基因的全流程。

- [scATAC-seq](./tutorials/scATACseq.ipynb)
- [scRNA-seq](./tutorials/scRNAseq.ipynb)
- [spatial_scATAC-seq](./tutorials/spatial_scATACseq.ipynb)
- [spatial_scRNA-seq](./tutorials/spatial_scRNAseq.ipynb)