"""The 9 integrity rules. Each is a pure function Index -> list[finding].

No natural-language recognition anywhere: every rule is float arithmetic or
set/membership logic over a closed input (the design rule). Iteration
order is always over an explicitly id-sorted sequence -- never a bare
`set`/`dict.keys()` walk -- so output is stable across PYTHONHASHSEED.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .io import Index

Finding = Dict[str, Any]


def _finding(rule_id: str, severity: str, ids, evidence, message: str) -> Finding:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "ids": sorted(ids),
        "evidence": evidence,
        "message": message,
    }


def rule_mixed_normalization(index: Index) -> List[Finding]:
    """R1 -- some vectors unit-norm, some not, under cosine/IP.

    METRIC-AWARE: silent under L2, where an unnormalized index is correct.
    This is the rule most likely to false-fire on healthy data, so the
    metric check covers cosine AND ip, not half the class.
    """
    if index.manifest.build_metric not in ("cosine", "ip"):
        return []
    tol = index.manifest.normalization_tolerance
    recs = sorted(index.vectors, key=lambda r: r.id)
    deviations = {}
    for r in recs:
        if not np.isfinite(r.vector).all():
            continue  # non-finite vectors are rule 3's territory, not rule 1's
        norm = float(np.linalg.norm(r.vector))
        deviations[r.id] = abs(norm - 1.0)
    normalized_ids = sorted(i for i, d in deviations.items() if d <= tol)
    unnormalized_ids = sorted(i for i, d in deviations.items() if d > tol)
    if not normalized_ids or not unnormalized_ids:
        return []  # every finite vector agrees -> not mixed
    offenders = unnormalized_ids if len(unnormalized_ids) <= len(normalized_ids) else normalized_ids
    evidence = {
        "tolerance": tol,
        "normalized_count": len(normalized_ids),
        "unnormalized_count": len(unnormalized_ids),
        "measured_norm_deviation": {i: deviations[i] for i in offenders},
    }
    return [
        _finding(
            "mixed_normalization",
            "warning",
            offenders,
            evidence,
            f"index mixes normalized and unnormalized vectors under {index.manifest.build_metric} metric",
        )
    ]


def rule_dimension_mismatch(index: Index) -> List[Finding]:
    """R2 -- a vector whose length differs from the declared index dimension."""
    dim = index.manifest.dimension
    ev: Dict[str, int] = {}
    for r in sorted(index.vectors, key=lambda r: r.id):
        if r.vector.shape[0] != dim:
            ev[r.id] = int(r.vector.shape[0])
    if not ev:
        return []
    return [
        _finding(
            "dimension_mismatch",
            "error",
            list(ev),
            {"expected_dimension": dim, "measured_dimension": ev},
            f"vector length differs from declared dimension {dim}",
        )
    ]


def rule_nonfinite(index: Index) -> List[Finding]:
    """R3 -- NaN or Inf, which poison every comparison they touch."""
    ev: Dict[str, Any] = {}
    for r in sorted(index.vectors, key=lambda r: r.id):
        mask = ~np.isfinite(r.vector)
        if mask.any():
            ev[r.id] = {"nonfinite_positions": [int(i) for i in np.nonzero(mask)[0]]}
    if not ev:
        return []
    return [
        _finding("nonfinite_values", "error", list(ev), ev, "vector contains NaN or Inf"),
    ]


def rule_zero_vector(index: Index) -> List[Finding]:
    """R4 -- zero / degenerate vectors, undefined direction under cosine.

    Flagged regardless of declared metric: a truly zero vector carries no
    directional information under cosine/IP, and under L2 is almost always
    an upstream embedding failure rather than a legitimate data point --
    documented as a design decision in README.md.
    """
    ev: Dict[str, float] = {}
    for r in sorted(index.vectors, key=lambda r: r.id):
        if not np.isfinite(r.vector).all():
            continue  # rule 3's territory
        if float(np.linalg.norm(r.vector)) == 0.0:
            ev[r.id] = 0.0
    if not ev:
        return []
    return [
        _finding(
            "zero_vector",
            "warning",
            list(ev),
            {"measured_norm": ev},
            "zero vector has undefined direction",
        )
    ]


def rule_duplicate_ids(index: Index) -> List[Finding]:
    """R5 -- the same id appearing twice (in vectors and/or mapping)."""
    vec_counts = Counter(r.id for r in index.vectors)
    map_counts = Counter(m.id for m in index.mapping)
    ev: Dict[str, Dict[str, int]] = {}
    for i in sorted(vec_counts):
        if vec_counts[i] > 1:
            ev.setdefault(i, {})["vectors_count"] = vec_counts[i]
    for i in sorted(map_counts):
        if map_counts[i] > 1:
            ev.setdefault(i, {})["mapping_count"] = map_counts[i]
    if not ev:
        return []
    return [
        _finding(
            "duplicate_ids",
            "error",
            list(ev),
            ev,
            "id appears more than once in vectors and/or mapping",
        )
    ]


def rule_orphaned_vectors(index: Index) -> List[Finding]:
    """R6 -- an id with no entry in the source mapping."""
    vec_ids = sorted(set(r.id for r in index.vectors))
    map_ids = set(m.id for m in index.mapping)
    offenders = [i for i in vec_ids if i not in map_ids]
    if not offenders:
        return []
    return [
        _finding(
            "orphaned_vectors",
            "error",
            offenders,
            {"mapping_entry_count": len(map_ids)},
            "vector id has no entry in the source mapping",
        )
    ]


def rule_stale_references(index: Index) -> List[Finding]:
    """R7 -- a mapping entry pointing at a source that no longer exists."""
    ev: Dict[str, str] = {}
    for m in sorted(index.mapping, key=lambda m: (m.id, m.line)):
        full = os.path.join(index.corpus_root, m.source)
        if not os.path.isfile(full):
            ev[m.id] = m.source
    if not ev:
        return []
    return [
        _finding(
            "stale_reference",
            "error",
            list(ev),
            {"missing_source": ev},
            "mapping entry points at a source file that does not exist",
        )
    ]


def rule_duplicate_vectors(index: Index) -> List[Finding]:
    """R8 -- near-identical embeddings under distinct ids, inflating recall@k.

    Near-duplicate is measured by cosine similarity regardless of the
    declared metric (a generic "same direction" detector, documented and
    reported with the measured value, never silently rounded).

    Defined over DISTINCT ids: a candidate pair whose two ids are equal
    (the same id string appears twice -- rule 5's territory) is excluded
    from pairing itself, not filtered after the fact, so no a==b self-pair
    can ever reach evidence.pairs even when a duplicated id's two rows
    happen to also be near-identical vectors.
    """
    threshold = index.manifest.duplicate_threshold
    recs = sorted(index.vectors, key=lambda r: r.id)
    usable = [r for r in recs if np.isfinite(r.vector).all() and float(np.linalg.norm(r.vector)) > 0.0]
    pairs = []
    n = len(usable)
    for i in range(n):
        vi = usable[i].vector
        ni = float(np.linalg.norm(vi))
        for j in range(i + 1, n):
            vj = usable[j].vector
            if usable[j].id == usable[i].id:
                continue  # rule 8 is defined over DISTINCT ids; a repeated
                # id string (rule 5's territory) must never produce an
                # a==b self-pair here, regardless of how similar the two
                # rows' vectors are
            if vi.shape[0] != vj.shape[0]:
                continue  # rule 2's territory
            nj = float(np.linalg.norm(vj))
            denom = ni * nj
            if denom == 0:
                continue
            cos = float(np.dot(vi, vj) / denom)
            if cos >= threshold:
                pairs.append((usable[i].id, usable[j].id, cos))
    if not pairs:
        return []
    pairs.sort()
    offenders = sorted(set(p[0] for p in pairs) | set(p[1] for p in pairs))
    ev = {
        "threshold": threshold,
        "pairs": [{"a": a, "b": b, "cosine_similarity": c} for a, b, c in pairs],
    }
    return [
        _finding(
            "duplicate_vectors",
            "warning",
            offenders,
            ev,
            "near-identical embeddings under distinct ids inflate recall@k",
        )
    ]


def rule_metric_mismatch(index: Index) -> List[Finding]:
    """R9 -- the metric declared at build time differs from query time.

    Index-level, not id-specific -> ids[] is empty (vacuously satisfies the
    "every id resolves in the input" contract, documented in README.md).
    """
    if index.manifest.build_metric == index.manifest.query_metric:
        return []
    return [
        _finding(
            "metric_mismatch",
            "error",
            [],
            {"build_metric": index.manifest.build_metric, "query_metric": index.manifest.query_metric},
            "build-time metric differs from query-time metric",
        )
    ]


ALL_RULES = [
    rule_mixed_normalization,
    rule_dimension_mismatch,
    rule_nonfinite,
    rule_zero_vector,
    rule_duplicate_ids,
    rule_orphaned_vectors,
    rule_stale_references,
    rule_duplicate_vectors,
    rule_metric_mismatch,
]
