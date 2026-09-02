"""Planted-defect corpus generator for veclint.

From a FIXED SEED, deterministically produces one fixture directory per
defect class plus clean controls -- reproducible from a fresh clone, no
external service, no download, no network. Run directly to write fixtures
to disk, or import `generate_all()` to build them into a temp dir for tests.

    python gen_corpus.py --out fixtures

Each fixture is a self-contained index directory:
    manifest.json, vectors.jsonl (or .npy + .ids.json), mapping.jsonl, corpus/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_SEED = 20240902
DIM = 8
TOL = 1e-3
DUP_THRESHOLD = 0.999


def _write_index(
    base_dir: str,
    manifest: dict,
    vectors: List[Tuple[str, list]],
    mapping: List[Tuple[str, str]],
    corpus_files: Dict[str, str],
) -> None:
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(base_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    fmt = manifest["format"]
    if fmt == "veclint-npy-v1":
        ids = [vid for vid, _ in vectors]
        arr = np.array([vec for _, vec in vectors], dtype=np.float64)
        np.save(os.path.join(base_dir, "vectors.npy"), arr)
        with open(os.path.join(base_dir, "vectors.ids.json"), "w", encoding="utf-8") as f:
            json.dump(ids, f)
    else:
        # veclint-jsonl-v1, and also the "unsupported format" fixture -- the
        # tool must refuse the latter by declared-format lookup, never by
        # guess-parsing the file it happens to find.
        with open(os.path.join(base_dir, "vectors.jsonl"), "w", encoding="utf-8") as f:
            for vid, vec in vectors:
                f.write(json.dumps({"id": vid, "vector": vec}) + "\n")

    with open(os.path.join(base_dir, "mapping.jsonl"), "w", encoding="utf-8") as f:
        for mid, source in mapping:
            f.write(json.dumps({"id": mid, "source": source}) + "\n")

    corpus_dir = os.path.join(base_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    for name, content in corpus_files.items():
        with open(os.path.join(corpus_dir, name), "w", encoding="utf-8") as f:
            f.write(content)


def _basic_manifest(build_metric: str, query_metric: str, fmt: str = "veclint-jsonl-v1", dim: int = DIM) -> dict:
    return {
        "format": fmt,
        "dimension": dim,
        "build_metric": build_metric,
        "query_metric": query_metric,
        "normalization_tolerance": TOL,
        "duplicate_threshold": DUP_THRESHOLD,
    }


def _ids(n: int, prefix: str = "id") -> List[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


def _unit_vectors(rng: np.random.RandomState, n: int, dim: int = DIM) -> List[np.ndarray]:
    out = []
    for _ in range(n):
        v = rng.normal(size=dim)
        v = v / np.linalg.norm(v)
        out.append(v)
    return out


def _mapping_and_corpus(ids: List[str]) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    mapping = [(i, f"{i}.txt") for i in ids]
    corpus = {f"{i}.txt": f"source content for {i}\n" for i in ids}
    return mapping, corpus


def make_clean(seed: int) -> dict:
    """Clean cosine index: unit-norm vectors, complete mapping, no dupes."""
    rng = np.random.RandomState(seed)
    n = 12
    ids = _ids(n)
    vecs = _unit_vectors(rng, n)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "cosine")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=(),
    )


def make_clean_l2(seed: int) -> dict:
    """Clean L2 index: deliberately UNNORMALIZED (varying norms). Metric-
    awareness gate: this must produce ZERO mixed_normalization findings."""
    rng = np.random.RandomState(seed)
    n = 12
    ids = _ids(n, prefix="l2id")
    vecs = []
    for _ in range(n):
        v = rng.normal(size=DIM) * rng.uniform(0.5, 5.0)
        vecs.append(v)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=(),
    )


def make_defect_mixed_norm(seed: int) -> dict:
    """R1: cosine metric, 7 unit-norm + 3 clearly-unnormalized (norm~3)."""
    rng = np.random.RandomState(seed)
    n = 10
    ids = _ids(n, prefix="mn")
    vecs = _unit_vectors(rng, n)
    offending = ids[7:10]
    for i in range(7, 10):
        vecs[i] = vecs[i] * 3.0
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "cosine")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("mixed_normalization",),
        expected_ids={"mixed_normalization": sorted(offending)},
    )


def make_defect_dim_mismatch(seed: int) -> dict:
    """R2: one vector has the wrong length. l2 metric to stay isolated
    from rule 1 (unnormalized vectors of any dim would otherwise also
    look "mixed" under cosine)."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="dm")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    bad_id = ids[3]
    vecs[3] = rng.normal(size=DIM + 3)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("dimension_mismatch",),
        expected_ids={"dimension_mismatch": [bad_id]},
    )


def make_defect_nonfinite(seed: int) -> dict:
    """R3: one vector has a NaN, another has an Inf. l2 metric to stay
    isolated from rule 1."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="nf")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    nan_id, inf_id = ids[2], ids[5]
    vecs[2] = vecs[2].copy()
    vecs[2][0] = float("nan")
    vecs[5] = vecs[5].copy()
    vecs[5][1] = float("inf")
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("nonfinite_values",),
        expected_ids={"nonfinite_values": sorted([nan_id, inf_id])},
    )


def make_defect_zero_vector(seed: int) -> dict:
    """R4: one vector is all zeros. l2 metric to stay isolated from rule 1."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="zv")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    zero_id = ids[4]
    vecs[4] = np.zeros(DIM)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("zero_vector",),
        expected_ids={"zero_vector": [zero_id]},
    )


def make_defect_dup_ids(seed: int) -> dict:
    """R5: one id appears twice in vectors.jsonl. l2 metric for isolation."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="di")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    dup_id = ids[1]
    all_ids = list(ids) + [dup_id]
    all_vecs = list(vecs) + [rng.normal(size=DIM)]
    mapping, corpus = _mapping_and_corpus(ids)  # mapping only has the id once
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(all_ids, [v.tolist() for v in all_vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("duplicate_ids",),
        expected_ids={"duplicate_ids": [dup_id]},
    )


def make_defect_orphaned(seed: int) -> dict:
    """R6: one vector id has no mapping entry. l2 metric for isolation."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="or")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    orphan_id = ids[6]
    mapped_ids = [i for i in ids if i != orphan_id]
    mapping, corpus = _mapping_and_corpus(mapped_ids)
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("orphaned_vectors",),
        expected_ids={"orphaned_vectors": [orphan_id]},
    )


def make_defect_stale_ref(seed: int) -> dict:
    """R7: one mapping entry points at a source file never written to
    corpus/. l2 metric for isolation."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="sr")
    vecs = [rng.normal(size=DIM) for _ in range(n)]
    stale_id = ids[5]
    mapping, corpus = _mapping_and_corpus(ids)
    # point the stale id's mapping entry at a file that is never created
    mapping = [(i, s if i != stale_id else "does_not_exist.txt") for i, s in mapping]
    manifest = _basic_manifest("l2", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("stale_reference",),
        expected_ids={"stale_reference": [stale_id]},
    )


def make_defect_dup_vectors(seed: int) -> dict:
    """R8: two vectors are near-identical (cosine sim >= threshold) under
    distinct ids. cosine metric, all unit-norm, to stay isolated from rule 1."""
    rng = np.random.RandomState(seed)
    n = 9
    ids = _ids(n, prefix="dv")
    vecs = _unit_vectors(rng, n)
    base_idx, dup_idx = 2, 7
    base = vecs[base_idx]
    sigma = 0.02
    for _ in range(20):
        noisy = base + rng.normal(size=DIM) * sigma
        noisy = noisy / np.linalg.norm(noisy)
        cos = float(np.dot(base, noisy))
        if DUP_THRESHOLD <= cos < 1.0:
            vecs[dup_idx] = noisy
            break
        sigma *= 0.5
    else:  # pragma: no cover - deterministic seed always converges
        vecs[dup_idx] = base.copy()
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "cosine")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("duplicate_vectors",),
        expected_ids={"duplicate_vectors": sorted([ids[base_idx], ids[dup_idx]])},
    )


def make_defect_metric_mismatch(seed: int) -> dict:
    """R9: build_metric != query_metric. Vectors kept unit-norm & cosine
    build-metric so rule 1 stays silent."""
    rng = np.random.RandomState(seed)
    n = 8
    ids = _ids(n, prefix="mm")
    vecs = _unit_vectors(rng, n)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "l2")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=("metric_mismatch",),
        expected_ids={"metric_mismatch": []},
    )


def make_defect_npy_format(seed: int) -> dict:
    """Supported alt format (veclint-npy-v1) clean control, to prove
    format coverage isn't jsonl-only."""
    rng = np.random.RandomState(seed)
    n = 10
    ids = _ids(n, prefix="npy")
    vecs = _unit_vectors(rng, n)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "cosine", fmt="veclint-npy-v1")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=(),
    )


def make_unsupported_format(seed: int) -> dict:
    """A manifest declaring a format outside SUPPORTED_FORMATS. The tool
    must refuse by declared-format lookup, not guess-parse the jsonl file
    that happens to sit next to it."""
    rng = np.random.RandomState(seed)
    n = 5
    ids = _ids(n, prefix="us")
    vecs = _unit_vectors(rng, n)
    mapping, corpus = _mapping_and_corpus(ids)
    manifest = _basic_manifest("cosine", "cosine", fmt="veclint-csv-v1")
    return dict(
        manifest=manifest,
        vectors=list(zip(ids, [v.tolist() for v in vecs])),
        mapping=mapping,
        corpus=corpus,
        expected_rule_ids=None,  # sentinel: this fixture is format-unsupported
    )


# name -> (builder, is_clean_control)
FIXTURE_BUILDERS = {
    "clean": (make_clean, True),
    "clean_l2": (make_clean_l2, True),
    "clean_npy_format": (make_defect_npy_format, True),
    "defect_mixed_norm": (make_defect_mixed_norm, False),
    "defect_dim_mismatch": (make_defect_dim_mismatch, False),
    "defect_nonfinite": (make_defect_nonfinite, False),
    "defect_zero_vector": (make_defect_zero_vector, False),
    "defect_dup_ids": (make_defect_dup_ids, False),
    "defect_orphaned": (make_defect_orphaned, False),
    "defect_stale_ref": (make_defect_stale_ref, False),
    "defect_dup_vectors": (make_defect_dup_vectors, False),
    "defect_metric_mismatch": (make_defect_metric_mismatch, False),
    "unsupported_format": (make_unsupported_format, None),
}


def generate_all(out_dir: str, seed: int = DEFAULT_SEED) -> Dict[str, dict]:
    """Writes every fixture under out_dir/<name>/ and returns name -> spec
    dict (including expected_rule_ids / expected_ids for test assertions)."""
    specs = {}
    for offset, (name, (builder, _clean)) in enumerate(sorted(FIXTURE_BUILDERS.items())):
        spec = builder(seed + offset)
        _write_index(
            os.path.join(out_dir, name),
            spec["manifest"],
            spec["vectors"],
            spec["mapping"],
            spec["corpus"],
        )
        specs[name] = spec
    return specs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate veclint's planted-defect fixture corpus.")
    parser.add_argument("--out", default="fixtures", help="output directory (default: fixtures)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    specs = generate_all(args.out, args.seed)
    print(f"wrote {len(specs)} fixtures to {args.out} (seed={args.seed})")
    for name in sorted(specs):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
