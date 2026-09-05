import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack


def csv_to_adata(data, celltype, celltype_key="celltype"):
    """Build an AnnData object from a cells-by-features table and labels."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data must be a non-empty cells-by-features table.")
    if data.index.has_duplicates:
        raise ValueError("Cell identifiers in data.index must be unique.")
    if data.columns.has_duplicates:
        raise ValueError("Feature identifiers in data.columns must be unique.")

    if isinstance(celltype, pd.DataFrame):
        if celltype.shape[1] != 1:
            raise ValueError("celltype DataFrame must contain exactly one column.")
        labels = celltype.iloc[:, 0]
    elif isinstance(celltype, pd.Series):
        labels = celltype
    else:
        labels = pd.Series(np.asarray(celltype).reshape(-1), index=data.index)

    if len(labels) != len(data):
        raise ValueError("celltype must contain one label for each row in data.")
    if isinstance(labels.index, pd.Index) and set(labels.index) == set(data.index):
        labels = labels.reindex(data.index)
    else:
        labels = pd.Series(labels.to_numpy(), index=data.index)
    if labels.isna().any():
        raise ValueError("celltype contains missing labels or unmatched cell identifiers.")

    numeric = data.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("data contains NaN or infinite values.")

    obs = pd.DataFrame({celltype_key: labels.to_numpy()}, index=data.index.astype(str))
    var = pd.DataFrame(index=data.columns.astype(str))
    return ad.AnnData(csr_matrix(numeric), obs=obs, var=var)


def seurat_csv_files_to_adata(
    expression_csv,
    celltype_csv,
    celltype_key="celltype",
    chunk_size=500,
):
    """Load a Seurat-style genes-by-cells CSV pair into a sparse AnnData object.

    The expression file must contain gene identifiers in its first column and
    cell identifiers in its header. The annotation file must contain cell
    identifiers in its first column and one cell-type column.
    """
    expression_path = Path(expression_csv)
    celltype_path = Path(celltype_csv)
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not expression_path.is_file():
        raise FileNotFoundError(f"Expression CSV not found: {expression_path}")
    if not celltype_path.is_file():
        raise FileNotFoundError(f"Cell-type CSV not found: {celltype_path}")

    annotations = pd.read_csv(celltype_path, index_col=0)
    if annotations.shape[1] != 1:
        raise ValueError("Cell-type CSV must contain exactly one annotation column.")
    if annotations.index.has_duplicates:
        raise ValueError("Cell identifiers in the cell-type CSV must be unique.")
    if annotations.isna().any().any():
        raise ValueError("Cell-type CSV contains missing annotations.")
    annotations.index = annotations.index.astype(str)
    annotations.columns = [celltype_key]

    expression_header = pd.read_csv(expression_path, index_col=0, nrows=0)
    cell_ids = expression_header.columns.astype(str)
    if cell_ids.has_duplicates:
        raise ValueError("Cell identifiers in the expression CSV must be unique.")
    missing = cell_ids.difference(annotations.index)
    extra = annotations.index.difference(cell_ids)
    if len(missing) or len(extra):
        raise ValueError(
            "Expression and annotation cell identifiers do not match "
            f"(missing annotations: {len(missing)}, extra annotations: {len(extra)})."
        )
    annotations = annotations.reindex(cell_ids)

    matrix_blocks = []
    feature_ids = []
    observed_features = set()
    reader = pd.read_csv(
        expression_path,
        index_col=0,
        dtype={cell_id: np.float32 for cell_id in cell_ids},
        chunksize=int(chunk_size),
    )
    for block in reader:
        block.index = block.index.astype(str)
        duplicated = [name for name in block.index if name in observed_features]
        if block.index.has_duplicates or duplicated:
            raise ValueError("Feature identifiers in the expression CSV must be unique.")
        values = block.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Expression CSV contains NaN or infinite values.")
        if values.size and values.min() < 0:
            raise ValueError("Expression CSV must contain non-negative values.")
        matrix_blocks.append(csr_matrix(values).T)
        feature_ids.extend(block.index.tolist())
        observed_features.update(block.index)

    if not matrix_blocks:
        raise ValueError("Expression CSV contains no features.")
    matrix = hstack(matrix_blocks, format="csr", dtype=np.float32)
    obs = annotations.copy()
    var = pd.DataFrame(index=pd.Index(feature_ids, name="feature"))
    adata = ad.AnnData(matrix, obs=obs, var=var)
    adata.uns["source_files"] = {
        "expression_csv": expression_path.name,
        "celltype_csv": celltype_path.name,
    }
    return adata
