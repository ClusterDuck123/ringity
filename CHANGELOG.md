# Changelog

All notable changes to ringity are documented in this file.
## 0.5a1 — 2026-10-19

### Changed
- **Requires Python ≥ 3.10** (was ≥ 3.8). pip on Python 3.8/3.9 will keep installing 0.5a0.
- `ring_score_from_sequence`: `flavour` must now match exactly. Previously any
  substring of `"geometric"` (e.g. `"geo"`) silently selected the geometric score.
- `pdiagram_from_network` now accepts `persistence` and `dim`, and passes extra
  keyword arguments through to the distance calculation.
- `pdiagram_from_anndata` / `ring_score_from_anndata` now pass `dim`, `metric` and
  `persistence` through instead of ignoring them.

### Removed
- `metric_params` argument from `ring_score_from_point_cloud`,
  `ring_score_from_network` and `vietoris_rips_from_point_cloud`.

### Fixed
- `ring_score_from_network` ignored `use_weights` and always used the default.
  **Results can change** if you passed `use_weights` explicitly.
- `distance_to_knn_graph` with `keep_weights` raised an error instead of building a
  weighted graph.
- The `er_fc_th` threshold no longer raises "Quantiles must be in [0, 1]" on small graphs.
- `annulus()` now passes `n_neighbors` through.
- NetworkX 3 compatibility: replaced the removed `from_/to_scipy_sparse_matrix`
  (affected weighted kNN graphs and `triangle_signature`).

### Development
- Adopted ruff linting via pre-commit; removed remaining `unittest` usage.

---

## 0.5a0 — 2026-06-14

### Fixed
- `resistance` distance metric: replace the removed SciPy sparse `.A` attribute with
  `.toarray()`, restoring resistance-based ring-score computation on current SciPy.
- Network construction edge case in the ring-score pipeline.

### Changed
- Migrated the test runner to `pytest` and added a GitHub Actions test workflow.
