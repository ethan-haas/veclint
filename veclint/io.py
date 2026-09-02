"""Input loading for veclint.

Coverage is a DECLARED table of supported index formats (SUPPORTED_FORMATS
below). Anything else is reported `unsupported`, stated plainly, and never
guess-parsed -- this is a deliberate design requirement, not an
afterthought.

An index directory holds:
  manifest.json    -- format id, dimension, build/query metric, tolerances
  vectors.<ext>     -- one of the formats in SUPPORTED_FORMATS
  mapping.jsonl     -- {"id": ..., "source": ...} per line
  corpus/           -- the actual source files mapping entries point at

All vectors are read from files. Nothing here touches the network, a
vector-DB service, an API key, or an embedding model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np

# The closed, declared set of index formats this tool understands. A format
# string outside this tuple is `unsupported` -- never parsed on a guess.
SUPPORTED_FORMATS = ("veclint-jsonl-v1", "veclint-npy-v1")

VALID_METRICS = ("cosine", "ip", "l2")


class MalformedInputError(Exception):
    """Structurally invalid input -> exit code 2."""


class UnsupportedFormatError(Exception):
    """manifest.json declares a format outside SUPPORTED_FORMATS -> exit code 2."""

    def __init__(self, fmt):
        self.fmt = fmt
        super().__init__(f"unsupported index format: {fmt!r}")


@dataclass(frozen=True)
class Manifest:
    format: str
    dimension: int
    build_metric: str
    query_metric: str
    normalization_tolerance: float
    duplicate_threshold: float


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: np.ndarray
    line: int  # 1-based position in the source file/array, for evidence


@dataclass(frozen=True)
class MappingRecord:
    id: str
    source: str
    line: int


@dataclass(frozen=True)
class Index:
    manifest: Manifest
    vectors: List[VectorRecord]
    mapping: List[MappingRecord]
    corpus_root: str


def load_manifest(index_dir: str) -> Manifest:
    path = os.path.join(index_dir, "manifest.json")
    if not os.path.isfile(path):
        raise MalformedInputError(f"missing manifest.json in {index_dir}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise MalformedInputError(f"manifest.json is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise MalformedInputError("manifest.json must be a JSON object")

    required = ("format", "dimension", "build_metric", "query_metric")
    for key in required:
        if key not in raw:
            raise MalformedInputError(f"manifest.json missing required key: {key}")

    fmt = raw["format"]
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(fmt)

    if raw["build_metric"] not in VALID_METRICS or raw["query_metric"] not in VALID_METRICS:
        raise MalformedInputError(f"manifest.json metric must be one of {VALID_METRICS}")

    dimension = raw["dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise MalformedInputError("manifest.json dimension must be a positive int")

    return Manifest(
        format=fmt,
        dimension=dimension,
        build_metric=raw["build_metric"],
        query_metric=raw["query_metric"],
        normalization_tolerance=float(raw.get("normalization_tolerance", 1e-3)),
        duplicate_threshold=float(raw.get("duplicate_threshold", 0.999)),
    )


def _load_vectors_jsonl(index_dir: str) -> List[VectorRecord]:
    path = os.path.join(index_dir, "vectors.jsonl")
    if not os.path.isfile(path):
        raise MalformedInputError(f"missing vectors.jsonl in {index_dir}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise MalformedInputError(f"vectors.jsonl:{lineno} not valid JSON: {e}") from e
            if "id" not in obj or "vector" not in obj:
                raise MalformedInputError(f"vectors.jsonl:{lineno} missing id/vector")
            vec = np.asarray(obj["vector"], dtype=np.float64)
            if vec.ndim != 1:
                raise MalformedInputError(f"vectors.jsonl:{lineno} vector must be 1-D")
            records.append(VectorRecord(id=str(obj["id"]), vector=vec, line=lineno))
    return records


def _load_vectors_npy(index_dir: str) -> List[VectorRecord]:
    npy_path = os.path.join(index_dir, "vectors.npy")
    ids_path = os.path.join(index_dir, "vectors.ids.json")
    if not os.path.isfile(npy_path):
        raise MalformedInputError(f"missing vectors.npy in {index_dir}")
    if not os.path.isfile(ids_path):
        raise MalformedInputError(f"missing vectors.ids.json in {index_dir}")
    arr = np.load(npy_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise MalformedInputError("vectors.ids.json must be a JSON array")
    if arr.ndim != 2:
        raise MalformedInputError("vectors.npy must be a 2-D array")
    if len(ids) != arr.shape[0]:
        raise MalformedInputError(
            f"vectors.ids.json length ({len(ids)}) != vectors.npy rows ({arr.shape[0]})"
        )
    records = []
    for i, vid in enumerate(ids):
        records.append(VectorRecord(id=str(vid), vector=arr[i].astype(np.float64), line=i + 1))
    return records


def _load_mapping_jsonl(index_dir: str) -> List[MappingRecord]:
    path = os.path.join(index_dir, "mapping.jsonl")
    if not os.path.isfile(path):
        raise MalformedInputError(f"missing mapping.jsonl in {index_dir}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise MalformedInputError(f"mapping.jsonl:{lineno} not valid JSON: {e}") from e
            if "id" not in obj or "source" not in obj:
                raise MalformedInputError(f"mapping.jsonl:{lineno} missing id/source")
            records.append(MappingRecord(id=str(obj["id"]), source=str(obj["source"]), line=lineno))
    return records


VECTOR_LOADERS = {
    "veclint-jsonl-v1": _load_vectors_jsonl,
    "veclint-npy-v1": _load_vectors_npy,
}


def load_index(index_dir: str) -> Index:
    manifest = load_manifest(index_dir)  # raises UnsupportedFormatError first, before touching anything else
    vectors = VECTOR_LOADERS[manifest.format](index_dir)
    mapping = _load_mapping_jsonl(index_dir)
    corpus_root = os.path.join(index_dir, "corpus")
    return Index(manifest=manifest, vectors=vectors, mapping=mapping, corpus_root=corpus_root)
