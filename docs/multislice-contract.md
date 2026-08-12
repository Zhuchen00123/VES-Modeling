# VES Modeling multi-slice contract (T-010)

This document is the cross-cutting public contract for the six vertical
slices (regression, forecasting, classification, optimization, ODE,
clustering, anomaly detection, graph).  Per-slice behavior contracts live in each slice's data
contract module; this page records the shared decisions and the frozen
API naming used by all slices.

## Slices and stable entry points

| Slice | Builder export | Search API | Apply API | Verified metrics |
|---|---|---|---|---|
| Regression | `build_regression_problem` | `run_regression_search` | `apply_regression_solution` | `rmse`, `mae` |
| Forecasting | `build_forecasting_problem` | `run_forecasting_search` | `apply_forecasting_solution` | `rmse`, `mae`, `smape` |
| Classification | `build_classification_problem` | `run_classification_search` | `apply_classification_solution` | `accuracy`, `macro_f1`, `log_loss`, `auroc`, `multiclass_brier`, `calibration_ece`, `confusion_*` |
| Optimization | `build_optimization_problem` | `run_optimization_search` | `apply_optimization_solution` | `max_bound_violation`, `max_constraint_violation`, `integrality_violation`, `objective` |
| ODE | `build_ode_problem` | `run_ode_search` | `apply_ode_solution` | `rmse`, `mae` (on hidden (t, y) points) |
| Clustering | `build_clustering_problem` | `run_clustering_search` | `apply_clustering_solution` | `ari`, `nmi`, `v_measure`, `silhouette` (optional) |
| Anomaly | `build_anomaly_problem` | `run_anomaly_search` | `apply_anomaly_solution` | `auroc`, `average_precision` (score) / `f1`, `balanced_accuracy` (label) |
| Graph | `build_graph_problem` | `run_graph_search` | `apply_graph_solution` | `total_weight`/`total_value` + violations (`shortest_path`/`max_flow`/`min_spanning_tree`) |

Every slice declares `API_SCHEMA_VERSION = "1.0"` and distinguishes itself
through `capabilities()["operations"]`; the version is shared because the
result/summary envelope shape is uniform across slices.

## Shared rules

- **AI can propose; it cannot grade itself.**  Candidate self-reported
  metrics (for example `claimed_rmse`, `claimed_accuracy`, `objective`,
  `feasibility`, `optimality`, `gap`) are ignored.  Judge/Search consume only
  host-computed evidence.
- **Apply never invents labels.**  When an apply operation has no host truth,
  the only success status is `produced_unverified`; apply results never
  contain verified metrics.  Optimization is the one exception in shape but
  not in status: because `problem.json` is the complete public instance, apply
  re-attaches host-recomputed *facts* (`feasible`, `objective`, violation
  residuals) while the status remains `produced_unverified` and global
  optimality is never claimed.
- **Host truth stays host.**  Hidden labels never enter prompts,
  candidate-visible directories, containers, public artifacts, logs, or
  records.  Provenance exposes only one-way digests (for example a 64-hex
  hash of labels/class order) and file hashes of public inputs.
- **Untrusted code runs only in Docker** with `--network none`,
  `--read-only`, `--cap-drop ALL`, non-root, no-new-privileges and
  per-file public whitelist mounts.  `trusted_code=True` is an explicit local
  opt-in used only for fixtures/tests.
- **Strict data contracts fail before execution.**  Duplicate headers, bad
  IDs, missing/extra rows, non-finite values, unknown fields, wrong shapes and
  wrong senses are rejected before any candidate runs.

## Frozen naming decision (T-009 review)

The optimization slice initially considered a separate
`verify_optimization_solution` entry point with `feasible_verified` /
`infeasible` / `invalid` statuses.  Final decision: **keep
`apply_optimization_solution`** with status `produced_unverified` plus
host-recomputed fact fields, so all four slices share the uniform
`run_*_search` + `apply_*_solution` API shape.  A separate verify entry point
would be speculative API until a real workflow needs it.

## Frozen semantics notes

- Optimization tolerance defaults to `1e-6`; a violation magnitude equal to
  the tolerance passes (`value > maximum` fails).
- Classification ECE uses 10 equal-width bins and the last bin covers
  `confidence == 1.0` (`min(int(confidence * n_bins), n_bins - 1)`); this is
  the documented boundary semantics of the `calibration_ece` metric.
- Optimization `best_solution.py` may still be written when `best_code`
  exists but the host judged it infeasible; the `status` field expresses this
  (`rejected` / `no_verified`), same as regression behavior.
- Clustering `silhouette` is computed on the public `test_features.csv`
  with the candidate's labels when computable (>=2 clusters and enough
  samples); otherwise it falls back to `0.0` and never blocks verification.
  ARI/NMI/V-measure are permutation-invariant, so candidate cluster names
  never need to match host reference names.
- Graph feasibility is enforced structurally in the artifact contract
  (invalid paths/flow/trees are rejected before verification); the violation
  observations (`path_violation`, `tree_violation`, capacity/conservation
  residuals) are audit fields kept in the evidence for transparency.
- Replay requires an absolute `PYTHONPATH` (`.deps:src:...`); relative paths
  fail under `LocalRegressionRunner` because the child process changes its
  working directory to the run directory.

## Architecture guard

No `UniversalTask`, `TaskRegistry`, or `SolverRegistry` exists.  Each slice
owns its problem assembly, context, verifier, data contract and stable API.
Shared leaf utilities (runner, hashing, run diagnostics) are reused without
building a universal framework.  A package-level capability declaration is
provided by each slice's `capabilities()` and the four top-level builder
exports.
