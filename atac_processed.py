import argparse
from pathlib import Path

import episcanpy as epi
import numpy as np
import scanpy as sc
import scipy
from sklearn.feature_extraction.text import TfidfTransformer


def _as_1d(values):
    if scipy.sparse.issparse(values):
        return values.toarray().reshape(-1)
    return np.asarray(values).reshape(-1)


def _validate_count_matrix(count_mat):
    matrix = scipy.sparse.csr_matrix(count_mat, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("count_mat must be a non-empty peaks-by-cells matrix.")
    if not np.isfinite(matrix.data).all():
        raise ValueError("count_mat contains NaN or infinite values.")
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError("count_mat must contain non-negative values.")
    return matrix


def _inverse_nonzero(values):
    values = np.asarray(values, dtype=np.float64)
    inverse = np.zeros_like(values)
    np.divide(1.0, values, out=inverse, where=values != 0)
    return inverse


# Perform TF-IDF (count_mat: peak*cell)
def tfidf1(count_mat):
    matrix = _validate_count_matrix(count_mat)
    cell_totals = _as_1d(matrix.sum(axis=0))
    peak_totals = _as_1d(matrix.sum(axis=1))
    term_frequency = matrix.multiply(_inverse_nonzero(cell_totals))
    inverse_document_frequency = np.log1p(
        matrix.shape[1] * _inverse_nonzero(peak_totals)
    )
    return term_frequency.multiply(inverse_document_frequency[:, None]).tocsr()


# Perform Signac TF-IDF (count_mat: peak*cell)
def tfidf2(count_mat):
    matrix = _validate_count_matrix(count_mat)
    cell_totals = _as_1d(matrix.sum(axis=0))
    peak_totals = _as_1d(matrix.sum(axis=1))
    term_frequency = matrix.multiply(_inverse_nonzero(cell_totals))
    inverse_document_frequency = matrix.shape[1] * _inverse_nonzero(peak_totals)
    signac = term_frequency.multiply((1e4 * inverse_document_frequency)[:, None]).tocsr()
    signac.data = np.log1p(signac.data)
    signac.eliminate_zeros()
    return signac


def tfidf3(count_mat):
    count_mat = _validate_count_matrix(count_mat)
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    return scipy.sparse.csr_matrix(tf_idf)


def preprocess_atac(adata, fpeak=0.05, run_lazy=True, copy_adata=True):
    """Binarize, feature-filter and TF-IDF transform an ATAC AnnData matrix."""
    if not 0 <= float(fpeak) <= 1:
        raise ValueError("fpeak must be between 0 and 1.")
    data = adata.copy() if copy_adata else adata
    epi.pp.binarize(data)
    epi.pp.filter_features(data, min_cells=max(1, int(np.ceil(fpeak * data.n_obs))))
    if data.n_vars == 0:
        raise ValueError("No ATAC features remain after filtering.")
    data.X = tfidf2(data.X.T).T.tocsr()
    if run_lazy:
        epi.pp.lazy(data)  # PCA, tSNE and UMAP
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess scATAC-seq/spatial ATAC-seq AnnData.")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--fpeak", type=float, default=0.05)
    parser.add_argument(
        "--lazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run epi.pp.lazy (use --no-lazy to skip PCA/tSNE/UMAP)",
    )
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = sc.read_h5ad(input_path)
    data = preprocess_atac(data, fpeak=args.fpeak, run_lazy=args.lazy)
    output_path = output_dir / f"{input_path.stem}_processed{input_path.suffix}"
    data.write(output_path)
    print(data)
    print(f"The processed data has been output to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
