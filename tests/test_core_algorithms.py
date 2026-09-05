from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CSFEATURES_N_JOBS", "1")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import atac_processed
from marker_utils import core_algorithms as core
from marker_utils.make_adata import csv_to_adata, seurat_csv_files_to_adata


def make_adata(*, dense: bool, spatial: bool = True) -> ad.AnnData:
    rng = np.random.default_rng(20260827)
    x = rng.poisson(1.0, size=(16, 8)).astype(np.float32)
    x[:8, :2] += 4
    x[8:, 2:4] += 4
    x[:, -1] = 0  # exercise numerical stability for an all-zero feature
    matrix = x if dense else sparse.csr_matrix(x)
    obs = pd.DataFrame(
        {"celltype": ["A"] * 8 + ["B"] * 8},
        index=[f"cell_{i}" for i in range(16)],
    )
    var = pd.DataFrame(index=[f"feature_{i}" for i in range(8)])
    adata = ad.AnnData(matrix, obs=obs, var=var)
    if spatial:
        adata.obsm["spatial"] = rng.normal(size=(16, 2)).astype(np.float32)
    return adata


class CoreAlgorithmsCompatibilityTests(unittest.TestCase):
    def assert_finite_results(self, info, processed):
        self.assertEqual(set(info), {"A", "B"})
        for cluster, frame in info.items():
            self.assertEqual(len(frame), processed.n_vars)
            self.assertTrue(np.isfinite(frame["EI"].to_numpy()).all(), cluster)
            self.assertIn(f"{cluster}_EI", processed.var)
        self.assertIn("correctX", processed.obsm)
        self.assertEqual(processed.uns["csfeatures"]["software_version"], "V1.0")
        self.assertEqual(processed.uns["csfeatures"]["package_version"], "1.0.0")

    def test_dense_and_sparse_anndata_are_supported(self):
        dense_info, dense_processed = core.getMarkersEI(
            adata=make_adata(dense=True), n_comps=3, n_neighbors=4
        )
        sparse_info, sparse_processed = core.getMarkersEI(
            adata=make_adata(dense=False), n_comps=3, n_neighbors=4
        )
        self.assert_finite_results(dense_info, dense_processed)
        self.assert_finite_results(sparse_info, sparse_processed)
        for cluster in ("A", "B"):
            np.testing.assert_allclose(
                dense_info[cluster]["EI"],
                sparse_info[cluster]["EI"],
                rtol=1e-5,
                atol=1e-7,
            )

    def test_spatial_accepts_object_and_h5ad_path(self):
        adata = make_adata(dense=False)
        object_info, object_processed = core.get_spatial_MarkersEI(
            adata, n_comps=3, n_neighbors=4
        )
        self.assert_finite_results(object_info, object_processed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "spatial.h5ad"
            adata.write_h5ad(path)
            path_info, path_processed = core.get_spatial_MarkersEI(
                path, n_comps=3, n_neighbors=4
            )
        self.assert_finite_results(path_info, path_processed)

    def test_input_validation_reports_actionable_errors(self):
        missing_label = make_adata(dense=True)
        del missing_label.obs["celltype"]
        with self.assertRaisesRegex(KeyError, "celltype"):
            core.getMarkersEI(adata=missing_label, n_comps=3, n_neighbors=4)

        negative = make_adata(dense=True)
        negative.X[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            core.getMarkersEI(adata=negative, n_comps=3, n_neighbors=4)

        missing_spatial = make_adata(dense=False, spatial=False)
        with self.assertRaisesRegex(KeyError, "spatial"):
            core.get_spatial_MarkersEI(missing_spatial, n_comps=3, n_neighbors=4)

    def test_small_anndata_bounds_pca_and_neighbor_parameters(self):
        rng = np.random.default_rng(42)
        x = rng.poisson(1.0, size=(6, 4)).astype(np.float32)
        x[:3, 0] += 3
        x[3:, 1] += 3
        adata = ad.AnnData(
            sparse.csr_matrix(x),
            obs=pd.DataFrame(
                {"group": ["A"] * 3 + ["B"] * 3},
                index=[f"small_{i}" for i in range(6)],
            ),
            var=pd.DataFrame(index=[f"feature_{i}" for i in range(4)]),
        )
        info, processed = core.getMarkersEI(
            adata=adata,
            celltype_key="group",
            n_comps=50,
            n_neighbors=30,
        )
        self.assert_finite_results(info, processed)

    def test_batch_input_returns_median_rank_consensus(self):
        adata = make_adata(dense=False)
        adata.obs["batch"] = (
            ["batch_1"] * 4
            + ["batch_2"] * 4
            + ["batch_1"] * 4
            + ["batch_2"] * 4
        )
        consensus, per_batch, processed_batches = core.getMarkersEI(
            adata=adata,
            batch_col="batch",
            n_comps=3,
            n_neighbors=3,
        )
        self.assertEqual(set(per_batch), {"batch_1", "batch_2"})
        self.assertEqual(set(processed_batches), {"batch_1", "batch_2"})
        self.assertEqual(set(consensus), {"A", "B"})
        for celltype, frame in consensus.items():
            self.assertEqual(len(frame), adata.n_vars)
            self.assertEqual(frame["ConsensusRank"].tolist(), list(range(1, adata.n_vars + 1)))
            self.assertTrue(frame["MedianRank"].is_monotonic_increasing, celltype)
            self.assertTrue((frame["BatchesUsed"] == 2).all(), celltype)

    def test_median_rank_aggregation_uses_mean_ei_as_final_tie_breaker(self):
        features = ["g1", "g2", "g3"]
        batch_results = {
            "b1": {
                "A": pd.DataFrame({"Features": features, "EI": [0.9, 0.8, 0.1]})
            },
            "b2": {
                "A": pd.DataFrame({"Features": features, "EI": [0.4, 0.95, 0.2]})
            },
        }
        result = core.aggregate_batch_marker_ranks(batch_results)["A"]
        self.assertEqual(result["Features"].tolist(), ["g2", "g1", "g3"])
        np.testing.assert_allclose(result["MedianRank"], [1.5, 1.5, 3.0])
        self.assertEqual(result["BatchesUsed"].tolist(), [2, 2, 2])


class DataPreparationCompatibilityTests(unittest.TestCase):
    def test_csv_to_adata_aligns_labels_by_cell_identifier(self):
        cells = ["cell_1", "cell_2", "cell_3", "cell_4"]
        data = pd.DataFrame(
            np.arange(12, dtype=np.float32).reshape(4, 3),
            index=cells,
            columns=["gene_1", "gene_2", "gene_3"],
        )
        labels = pd.Series(
            ["B", "B", "A", "A"],
            index=list(reversed(cells)),
            name="cluster",
        )
        adata = csv_to_adata(data, labels, celltype_key="group")
        self.assertTrue(sparse.isspmatrix_csr(adata.X))
        self.assertEqual(list(adata.obs_names), cells)
        self.assertEqual(list(adata.var_names), list(data.columns))
        self.assertEqual(list(adata.obs["group"]), ["A", "A", "B", "B"])

    def test_seurat_csv_pair_is_loaded_as_sparse_anndata(self):
        expression = pd.DataFrame(
            [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 1.0]],
            index=["gene_1", "gene_2"],
            columns=["cell_1", "cell_2", "cell_3", "cell_4"],
        )
        labels = pd.DataFrame(
            {"celltype": ["A", "A", "B", "B"]}, index=expression.columns
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            expression_path = Path(tmp_dir) / "expression.csv"
            labels_path = Path(tmp_dir) / "celltype.csv"
            expression.to_csv(expression_path)
            labels.to_csv(labels_path)
            adata = seurat_csv_files_to_adata(
                expression_path, labels_path, chunk_size=1
            )
        self.assertEqual(adata.shape, (4, 2))
        self.assertTrue(sparse.isspmatrix_csr(adata.X))
        self.assertEqual(list(adata.obs_names), list(expression.columns))
        self.assertEqual(list(adata.var_names), list(expression.index))
        np.testing.assert_allclose(adata.X.toarray(), expression.to_numpy().T)

    def test_tfidf_sparse_implementation_matches_original_formulas(self):
        counts = np.array(
            [[1, 0, 2, 1], [0, 3, 1, 0], [1, 1, 0, 2]], dtype=np.float64
        )
        term_frequency = counts / counts.sum(axis=0, keepdims=True)
        inverse_frequency = counts.shape[1] / counts.sum(axis=1, keepdims=True)
        expected_tfidf1 = term_frequency * np.log1p(inverse_frequency)
        expected_tfidf2 = np.log1p(1e4 * term_frequency * inverse_frequency)

        for candidate in (counts, sparse.csr_matrix(counts)):
            result1 = atac_processed.tfidf1(candidate)
            result2 = atac_processed.tfidf2(candidate)
            self.assertTrue(sparse.isspmatrix_csr(result1))
            self.assertTrue(sparse.isspmatrix_csr(result2))
            np.testing.assert_allclose(result1.toarray(), expected_tfidf1)
            np.testing.assert_allclose(result2.toarray(), expected_tfidf2)

    def test_atac_preprocessing_returns_sparse_copy(self):
        counts = sparse.csr_matrix(
            np.array(
                [
                    [1, 0, 2, 0],
                    [0, 1, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 1],
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                ],
                dtype=np.float32,
            )
        )
        source = ad.AnnData(counts.copy())
        processed = atac_processed.preprocess_atac(
            source, fpeak=0.1, run_lazy=False, copy_adata=True
        )
        self.assertTrue(sparse.isspmatrix_csr(processed.X))
        self.assertTrue(np.isfinite(processed.X.data).all())
        np.testing.assert_array_equal(source.X.toarray(), counts.toarray())


if __name__ == "__main__":
    unittest.main()
