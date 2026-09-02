# veclint

Offline integrity linter for a vector index and the corpus behind it. It
looks for the class of defect that degrades ranking **without raising an
error**: nothing crashes, nothing logs an exception, recall@k just quietly
gets a few points worse and someone blames the chunker.

No natural-language recognition anywhere. Every rule is float arithmetic or
set/membership logic over a closed input.

## Dependency (not stdlib-only)

**veclint requires `numpy`.** See `requirements.txt` (pinned `numpy==2.4.1`).
`pytest==8.4.2` (also pinned in requirements.txt) is a dev/test dependency
only, not required to run the CLI. Install with:

```
pip install -r requirements.txt
```

Everything is offline: no network call, no vector-DB service, no API key,
no embedding model, in any code path including the test suite. Vectors are
read from files on disk.

## Input: the declared index-format table

veclint does **not** guess a file format from its extension alone. An index
directory's `manifest.json` declares a `"format"` string, and that string
is looked up in a closed table. Anything outside the table is reported
`{"status": "unsupported", ...}` with exit code 2 -- stated plainly, never
parsed on a guess.

| `format` id | vectors file(s) | mapping file | notes |
|---|---|---|---|
| `veclint-jsonl-v1` | `vectors.jsonl` -- one `{"id": str, "vector": [float,...]}` per line | `mapping.jsonl` -- one `{"id": str, "source": str}` per line | default / reference format |
| `veclint-npy-v1` | `vectors.npy` (2-D float array, N x dim) + `vectors.ids.json` (JSON array of N ids, row-aligned) | `mapping.jsonl`, same as above | for numpy-native pipelines |

Any other `format` string (or a directory with no `manifest.json` at all,
or one that fails to parse) is refused, not guessed at.

An index directory also has a `corpus/` subdirectory holding the actual
source files that `mapping.jsonl` entries name (relative paths, resolved
under `corpus/`). This is what rule 7 (stale references) checks against.

`manifest.json` schema (all keys except the two tolerances are required):

```json
{
  "format": "veclint-jsonl-v1",
  "dimension": 8,
  "build_metric": "cosine",
  "query_metric": "cosine",
  "normalization_tolerance": 0.001,
  "duplicate_threshold": 0.999
}
```

`build_metric` / `query_metric` are each one of `cosine`, `ip`, `l2`.

## Output / verdict contract

A single deterministic JSON object on stdout (`sort_keys=True`, fixed
separators -- see "Determinism" below):

```json
{
  "status": "clean" | "findings" | "malformed" | "unsupported",
  "manifest": {...declared config, echoed back...},
  "counts": {"vectors": N, "mapping_entries": M, "findings": K},
  "findings": [
    {"rule_id": "...", "severity": "error"|"warning", "ids": [...], "evidence": {...}, "message": "..."}
  ]
}
```

Every id in `finding["ids"]` resolves in the supplied input (a vector id or
a mapping id that was actually present) -- a finding that can't point at
its own evidence is a bug, not a finding. `veclint.cli.lint()` asserts this
internally before returning. Rule 9 (metric mismatch) is index-level, not
id-specific, so its `ids` is `[]` -- vacuously satisfying the contract.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | clean -- zero findings |
| `1` | one or more findings |
| `2` | malformed input, unrecognized/unsupported index format, or CLI usage error |

## The 9 rules

| # | rule_id | severity | metric-aware? |
|---|---|---|---|
| 1 | `mixed_normalization` | warning | yes -- silent under `l2`; checks `build_metric` |
| 2 | `dimension_mismatch` | error | no |
| 3 | `nonfinite_values` | error | no |
| 4 | `zero_vector` | warning | no (see note below) |
| 5 | `duplicate_ids` | error | no |
| 6 | `orphaned_vectors` | error | no |
| 7 | `stale_reference` | error | no |
| 8 | `duplicate_vectors` | warning | no (cosine-similarity based regardless of declared metric) |
| 9 | `metric_mismatch` | error | n/a |

### Tolerances (declared, not magic; reported in the output)

- `normalization_tolerance` (default `1e-3`): a vector's norm within this
  of `1.0` counts as "normalized" for rule 1. Every offending id in a
  `mixed_normalization` finding is reported with its measured
  `abs(norm - 1.0)`, never silently rounded into or out of the verdict.
- `duplicate_threshold` (default `0.999`): cosine similarity at or above
  this between two distinct ids' vectors triggers rule 8. Every reported
  pair carries its measured `cosine_similarity`.

Both values are echoed back in `result["manifest"]` on every run, and both
are per-finding fields in the relevant rules' `evidence`.

## Design decisions worth stating outright

- **Rule 1's "under a cosine/IP metric" reads `build_metric`.** The index's
  vectors are properties of how it was *built*; `query_metric` disagreement
  is rule 9's job. Rule 1's metric-awareness check explicitly covers both
  `cosine` and `ip` -- landing on only half this class is the specific
  failure this rule is written to avoid.
- **Rule 1 is silent when NO vector is unit-norm.** It reports a *mix*.
  A mix is evidence that two build paths disagree about normalization,
  which is a pipeline bug; an index where every vector is consistently
  non-unit is a consistent state, and under true cosine the magnitude
  divides out anyway. So a uniformly unnormalized index reports clean,
  by design -- if your engine implements "cosine" as inner product over
  vectors it assumes are pre-normalized, that assumption is the thing to
  check, and rule 9 covers the metric disagreement case.
- **Rule 1 flags the numeric minority class**, not "unnormalized" by a
  fixed label -- if 9 of 10 vectors are unnormalized and 1 is unit-norm,
  the lone unit-norm vector is the offender. This matches the intent
  ("some are, some aren't") without hardcoding which direction is wrong.
- **Rule 4 (zero vectors) fires regardless of declared metric.** A
  genuinely all-zero vector carries no directional information under
  cosine/IP, and under L2 is essentially always an upstream embedding
  failure rather than a legitimate data point at the origin -- unlike rule
  1, there is no metric under which a zero vector is a *correct*, expected
  state, so no metric-awareness gate applies here.
- **Rule 8 (duplicate vectors) uses cosine similarity as the near-duplicate
  measure regardless of `build_metric`.** It answers "do these two ids
  point at the same content," which is a direction question independent of
  which metric the index was built under.
- **`unsupported` is a distinct status from `malformed`,** both mapping to
  exit code 2 (the "malformed input/usage" bucket). Unsupported means
  "I recognize this isn't a format I know" (declared table lookup failed);
  malformed means "this claims to be a format I know but is structurally
  broken" (missing file, bad JSON, missing required key).
- **Coverage-format detection is by the manifest's declared `format`
  string, not the file extensions present.** A directory could physically
  contain a `vectors.jsonl` next to a manifest declaring an unrecognized
  format, and veclint must still refuse rather than notice the jsonl file
  and parse it anyway -- see `gen_corpus.make_unsupported_format` and
  `tests/test_unsupported_and_malformed.py`.

## Determinism

- Output is `json.dumps(result, sort_keys=True, separators=(",", ":"))` --
  dict key order is normalized regardless of insertion order.
- Every rule builds its offending-id lists from an explicitly `sorted(...,
  key=lambda r: r.id)` walk, never a bare `set`/`dict.keys()` iteration
  (`PYTHONHASHSEED` randomizes `str` hashing, which randomizes plain `set`
  iteration order across processes -- this is why plain sets are only used
  for O(1) membership tests here, always converted through `sorted()`
  before they reach output).
- No rule sums floats over a set or dict; per-vector norms and pairwise dot
  products are each a single fixed-order numpy reduction, not an
  accumulation across a hash-ordered collection.
- Verified in `tests/test_determinism.py`: >= 3 (in one test, 4) real
  subprocesses with `PYTHONHASHSEED` set to different values, asserting
  byte-identical stdout, across both a clean fixture and fixtures that hit
  every set/dict-shaped rule (duplicate ids, mixed normalization, duplicate
  vectors).

## Usage

```
python -m veclint --index-dir path/to/index_dir
```

Example against a generated fixture:

```
python gen_corpus.py --out fixtures
python -m veclint --index-dir fixtures/defect_mixed_norm
```

## Generating the planted-defect corpus

```
python gen_corpus.py --out fixtures --seed 20240902
```

Deterministic from the fixed seed (default `20240902`), no network, no
external service, reproducible from a fresh clone. Writes 13 fixture
directories: 3 clean controls (`clean` under cosine, `clean_l2`
deliberately unnormalized under L2 for the metric-awareness gate,
`clean_npy_format` exercising the second supported format), 9 single-defect
fixtures (one per rule), and 1 `unsupported_format` fixture.

The test suite does **not** depend on fixtures being pre-generated on disk;
`tests/conftest.py` regenerates the full corpus into a session-scoped temp
directory at test-collection time, so `pytest` alone is sufficient on a
fresh clone.

## Running the tests

```
pip install -r requirements.txt
pytest
```

What `tests/` covers (an independent adversarial review of the running
CLI is a separate exercise, not part of this suite):

- `test_rules.py` -- gate 1 (planted-defect corpus, exact rule_id + ids),
  gate 2 (clean fixtures produce zero findings), gate 3 (metric-awareness:
  unnormalized L2 stays silent, and separately, mixed-norm under `ip`
  fires -- proving the fix covers both halves of the cosine/IP class), gate
  4 (tolerance/threshold declared + reported with measured distance), and
  the verdict contract's id-resolution discipline.
- `test_determinism.py` -- gate 5, real subprocesses, differing
  `PYTHONHASHSEED`, byte-identical stdout.
- `test_positive_control.py` -- gate 6, mutate a rule (drop it from
  `ALL_RULES`, or loosen a tolerance), confirm the previously-caught defect
  now slips through -- proving the suite depends on the code, not vacuous.
- `test_unsupported_and_malformed.py` -- unsupported-format refusal and
  malformed-input exit code 2, distinct from "clean" and "findings".
- `test_corpus_metrics.py` -- `detection_rate`, `false_flag_rate`,
  `unsupported_rate` computed and reported **separately**, never averaged
  or blended into one number.

## Known limitations

- Rule 8's O(n^2) pairwise cosine-similarity scan is fine for the fixture
  sizes here (single or low double digits of vectors per fixture) but is
  not the algorithm to point at a production-scale index; a real deployment
  would want an ANN-based candidate-pair prefilter before the exact cosine
  check. Out of scope for this build, whose fixtures are deliberately
  synthetic and seed-generated rather than production-scale.
- `dimension` in `manifest.json` is a single declared int for the whole
  index; there is no per-shard or per-segment dimension override in this
  version.
