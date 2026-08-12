"""Candidate diagnostic persistence for the Monte Carlo API.

Monte Carlo reuses the Regression run.json writer (same run-tree layout), so
this module is a thin re-export and never duplicates the persistence logic.
"""

from __future__ import annotations

from ves_modeling.regression.diagnostics import (
    APPLY_STATUSES,
    CANDIDATE_STATUSES,
    detect_artifact_path,
    detect_solution_path,
    sha256_text,
    write_run_json,
)

__all__ = [
    "APPLY_STATUSES",
    "CANDIDATE_STATUSES",
    "detect_artifact_path",
    "detect_solution_path",
    "sha256_text",
    "write_run_json",
]
