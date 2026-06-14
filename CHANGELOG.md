# Changelog

All notable changes to ringity are documented in this file.

## 0.5a0 — 2026-06-14

### Fixed
- `resistance` distance metric: replace the removed SciPy sparse `.A` attribute with
  `.toarray()`, restoring resistance-based ring-score computation on current SciPy.
- Network construction edge case in the ring-score pipeline.

### Changed
- Migrated the test runner to `pytest` and added a GitHub Actions test workflow.
