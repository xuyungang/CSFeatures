# Changelog

## 1.0.0

CSFeatures V1.0 retains the published Expression Index (EI) workflow and
formulas while improving compatibility, robustness, and reproducibility.

### Added

- Software and package version identifiers.
- Exact CPython and dependency specifications in `environment.yml`,
  `requirements.txt`, and `pyproject.toml`.
- In-memory `AnnData` and `.h5ad` path input for single-cell and spatial entry
  points.
- Dense and sparse matrix compatibility.
- Configurable `celltype_key`, `spatial_key`, random seed, and copy behavior.
- Independent within-batch analysis and median-rank consensus aggregation.
- CSV-to-`AnnData` conversion with identifier alignment and chunked sparse
  loading.
- Sparse-preserving ATAC preprocessing callable from Python or the command line.
- Packaging metadata that includes both `marker_utils` and the
  `atac_processed` module.
- Input validation, small-dataset parameter bounds, run metadata, regression
  tests, and a V1.0 user notebook.

### Changed

- Neighbor indices and similarities used by the smoothness calculation are
  precomputed once per cell population, avoiding repeated graph-value
  construction without changing the smoothness definition or EI formula.
- Inputs are copied by default so an in-memory `AnnData` object is not modified
  unless `copy_adata=False` is selected.

### Compatibility note

- The four original tutorial notebooks and the MIT license are retained.
- V1.0 is validated with CPython 3.10.14 and the exact dependency versions in
  the environment files.
