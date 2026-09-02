"""veclint CLI: offline integrity linter for a vector index + its corpus.

Exit codes: 0 clean, 1 findings, 2 malformed input / unsupported format /
usage error. Output is a single deterministic JSON object on stdout
(sort_keys, fixed separators) so byte-identical output is achievable across
processes regardless of PYTHONHASHSEED.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Tuple

from .io import Index, MalformedInputError, SUPPORTED_FORMATS, UnsupportedFormatError, load_index
from .rules import ALL_RULES


def _all_known_ids(index: Index):
    ids = set(r.id for r in index.vectors)
    ids.update(m.id for m in index.mapping)
    return ids


def lint(index_dir: str) -> Tuple[dict, int]:
    """Run all rules over the index at index_dir. Returns (result, exit_code)."""
    try:
        index = load_index(index_dir)
    except UnsupportedFormatError as e:
        result = {
            "status": "unsupported",
            "format": e.fmt,
            "supported_formats": sorted(SUPPORTED_FORMATS),
            "findings": [],
        }
        return result, 2
    except MalformedInputError as e:
        result = {"status": "malformed", "error": str(e), "findings": []}
        return result, 2

    findings = []
    for rule_fn in ALL_RULES:
        findings.extend(rule_fn(index))
    findings.sort(key=lambda f: (f["rule_id"], f["ids"]))

    known_ids = _all_known_ids(index)
    for f in findings:
        for i in f["ids"]:
            if i not in known_ids:
                # A finding that cannot point at its own evidence is a bug,
                # not a finding (the verdict contract) -- fail loud.
                raise AssertionError(
                    f"internal error: finding {f['rule_id']} references unresolved id {i!r}"
                )

    result = {
        "status": "clean" if not findings else "findings",
        "manifest": {
            "format": index.manifest.format,
            "dimension": index.manifest.dimension,
            "build_metric": index.manifest.build_metric,
            "query_metric": index.manifest.query_metric,
            "normalization_tolerance": index.manifest.normalization_tolerance,
            "duplicate_threshold": index.manifest.duplicate_threshold,
        },
        "counts": {
            "vectors": len(index.vectors),
            "mapping_entries": len(index.mapping),
            "findings": len(findings),
        },
        "findings": findings,
    }
    exit_code = 1 if findings else 0
    return result, exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="veclint", description=__doc__)
    parser.add_argument(
        "--index-dir",
        required=True,
        help="directory holding manifest.json, vectors.*, mapping.jsonl, corpus/",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse already printed usage to stderr; normalize to exit code 2
        return 2 if e.code != 0 else 0

    try:
        result, exit_code = lint(args.index_dir)
    except Exception as e:  # pragma: no cover - defensive: unexpected loader crash
        result = {"status": "malformed", "error": f"unexpected error: {e}", "findings": []}
        exit_code = 2

    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
