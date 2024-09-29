# CSFeatures: Identification of Cell-Type-Specific Differential Features in Single-Cell and Spatial Omics Data

## Introduction
CSFeatures is a tool designed to identify differential genes or differential accessibility regions in single-cell and spatial data.

## Installation
It is recommended to create a new virtual environment using conda to run this project.

```
conda create -n CSFeatures python=3.10
conda activate CSFeatures
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
The data to be prepared includes an expression matrix for scRNA-seq or scATAC-seq (with rows as cells and columns as genes) and the corresponding cell types. For spatial data, spatial coordinates must also be provided. Spatial information can be extracted using methods such as STAGATE, SpatialPCA, or SpaGCN.

In Python, you need to provide an AnnData object as input. AnnData is a class designed to store single-cell data.

The provided AnnData format should meet the following requirements:
- The shape of AnnData should be (number of cells, number of genes or regions).
- The AnnData.obs must contain a 'celltype' field to provide cell classifications.
- If using spatial scRNA-seq/scATAC-seq data, the AnnData.obsm should include a 'spatial' field to provide spatial information.

The following is an example of scRNA-seq/scATAC-seq data:

```bash
AnnData object with n_obs × n_vars = 1000 × 2000
    obs: 'celltype'
```
The following is an example of spatial scRNA-seq/scATAC-seq data:

```bash
AnnData object with n_obs × n_vars = 1000 × 2000
    obs: 'celltype'
    obsm: 'spatial'
```

### 2. Find Differential Genes/Peaks

This section primarily utilizes the `getMarkersEI` and `get_spatial_MarkersEI` functions.

## `getMarkersEI` Function

This function processes single-cell gene expression data contained in an `AnnData` object and identifies marker genes based on various statistical methods. The function performs the following tasks:
- Dimensionality reduction
- Similarity graph construction
- Expression value adjustment
- Calculation of gene information metrics specified in `method_list`

### Parameters

- **`adata` (str)**:  
  An `AnnData` object containing the single-cell gene expression matrix. Its `.X` attribute should be a 2D array with rows corresponding to cells and columns to genes.

- **`n_comps` (int, optional)**:  
  The number of principal components to calculate in PCA. The default value is 50.

- **`n_neighbors` (int, optional)**:  
  The number of nearest neighbors used to build the k-nearest neighbors graph. The default value is 30.

- **`metric` (str, optional)**:  
  The distance metric used to compute cell similarities. The default is `'euclidean'`.

- **`method_list` (list, optional)**:  
  A list of functions used to compute various gene statistics. The default includes:

  - `calculate_mean_and_var`
  - `calculate_smoothness`
  - `calculate_V`
  - `calculate_prop`
  - `calculate_local_mean_max`
  - `calculate_EI`

  These functions should be defined elsewhere and are used to calculate different metrics for gene expression analysis.

### Return Values

- **`info`**:  
  A data structure (e.g., `DataFrame`) containing gene information metrics calculated based on the methods in `method_list`.

- **`adata`**:  
  The `adata` object with updated EI values.

## `get_spatial_MarkersEI` Function

This function processes single-cell gene expression data containing spatial information and identifies spatial marker genes based on various statistical methods. It performs dimensionality reduction, constructs a similarity matrix, adjusts expression values, and calculates gene information metrics specified in the `method_list` through the following steps.

### Parameters

- **`adata` (AnnData)**:  
  An `AnnData` object containing the single-cell gene expression matrix. Its `.X` attribute should be a 2D array with rows corresponding to cells and columns to genes.

- **`n_comps` (int, optional)**:  
  The number of principal components to calculate in PCA. The default value is 50.

- **`n_neighbors` (int, optional)**:  
  The number of nearest neighbors used to build the k-nearest neighbors graph. The default value is 30.

- **`metric` (str, optional)**:  
  The distance metric used to compute cell similarities. The default is `'euclidean'`.

- **`spatial_key` (str, optional)**:  
  The key used to specify spatial information. The default is `'spatial'`.

- **`method_list` (list, optional)**:  
  A list of functions used to compute various gene statistics. The default includes:

  - `calculate_mean_and_var`
  - `calculate_smoothness`
  - `calculate_V`
  - `calculate_prop`
  - `calculate_local_mean_max`
  - `calculate_EI`

### Return Values

- **`info`**:  
  Data containing gene information metrics calculated based on the methods in `method_list`.

- **`adata`**:  
  The `adata` object with updated EI values.

This repository provides four datasets on Google Drive: [scATAC-seq](https://drive.google.com/file/d/1mXGWKpOMR4I-mqhyAIQ_UFV6VbHwizdh/view?usp=drive_link), [scRNA-seq](https://drive.google.com/file/d/1LWOnXLHYn8W6GFQ2NTfi84JyY9B4XGOK/view?usp=drive_link), [spatial scATAC-seq](https://drive.google.com/file/d/1w7oxnwR_Nma5uTm0yOf4I2O44tGC5Dif/view?usp=drive_link), and [spatial scRNA-seq](https://drive.google.com/file/d/1U3_0FIBEcTLzTiAHQG00sMNSLvq7lFtl/view?usp=drive_link). The following sections will demonstrate the full workflow for identifying differential genes using this tool with these four datasets as examples.

- [scATAC-seq](./tutorials/scATAC-seq.ipynb)
- [scRNA-seq](./tutorials/scRNA-seq.ipynb)
- [spatial_scATAC-seq](./tutorials/spatial_ATAC-seq.ipynb)
- [spatial_scRNA-seq](./tutorials/spatial_RNA-seq.ipynb)

