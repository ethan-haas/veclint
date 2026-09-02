"""Acceptance-gate 1 & the id-resolution / evidence-discipline check:
each planted defect class is detected with the correct rule_id and the
exact offending ids, and every id in every finding resolves in the input."""
import pytest

from veclint.cli import lint


def _defect_names(specs):
    return sorted(name for name, spec in specs.items() if spec.get("expected_rule_ids"))


@pytest.mark.parametrize(
    "name",
    [
        "defect_mixed_norm",
        "defect_dim_mismatch",
        "defect_nonfinite",
        "defect_zero_vector",
        "defect_dup_ids",
        "defect_orphaned",
        "defect_stale_ref",
        "defect_dup_vectors",
        "defect_metric_mismatch",
    ],
)
def test_planted_defect_detected_with_exact_ids(name, specs, dirs):
    spec = specs[name]
    result, exit_code = lint(dirs[name])
    assert exit_code == 1, f"{name}: expected exit 1 (findings), got {exit_code}: {result}"

    got_rule_ids = sorted(f["rule_id"] for f in result["findings"])
    assert got_rule_ids == sorted(spec["expected_rule_ids"]), (
        f"{name}: expected rules {spec['expected_rule_ids']}, got {got_rule_ids}"
    )

    for rule_id, expected_ids in spec["expected_ids"].items():
        matches = [f for f in result["findings"] if f["rule_id"] == rule_id]
        assert len(matches) == 1, f"{name}: expected exactly one {rule_id} finding"
        assert matches[0]["ids"] == sorted(expected_ids), (
            f"{name}/{rule_id}: expected ids {sorted(expected_ids)}, got {matches[0]['ids']}"
        )


@pytest.mark.parametrize("name", ["clean", "clean_l2", "clean_npy_format"])
def test_clean_fixture_zero_findings(name, dirs):
    """Gate 2 (two-sided): a clean index must produce ZERO findings."""
    result, exit_code = lint(dirs[name])
    assert exit_code == 0, f"{name}: expected clean exit 0, got {exit_code}: {result}"
    assert result["findings"] == []
    assert result["status"] == "clean"


def test_metric_awareness_gate_l2_unnormalized_is_silent(dirs):
    """Gate 3: a correct, UNNORMALIZED L2 index must produce zero
    normalization findings -- the over-rejection control."""
    result, _ = lint(dirs["clean_l2"])
    norm_findings = [f for f in result["findings"] if f["rule_id"] == "mixed_normalization"]
    assert norm_findings == []


def test_metric_awareness_covers_both_cosine_and_ip():
    """Rule 1's metric-awareness must cover cosine AND ip, not half the
    class -- construct a mixed-norm fixture directly under ip and confirm
    it still fires -- landing on only half this class is the explicit
    failure mode this rule guards against."""
    import numpy as np

    import gen_corpus

    rng = np.random.RandomState(999)
    n = 6
    ids = [f"ipid{i:03d}" for i in range(n)]
    vecs = [(v / np.linalg.norm(v)).tolist() for v in (rng.normal(size=8) for _ in range(n - 2))]
    vecs += [(rng.normal(size=8) * 4.0).tolist() for _ in range(2)]  # 2 clearly unnormalized
    mapping = [(i, f"{i}.txt") for i in ids]
    corpus = {f"{i}.txt": "x" for i in ids}
    manifest = gen_corpus._basic_manifest("ip", "ip")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        gen_corpus._write_index(tmp, manifest, list(zip(ids, vecs)), mapping, corpus)
        result, exit_code = lint(tmp)

    assert exit_code == 1
    norm_findings = [f for f in result["findings"] if f["rule_id"] == "mixed_normalization"]
    assert len(norm_findings) == 1
    assert set(norm_findings[0]["ids"]) == set(ids[-2:])


def test_tolerance_and_threshold_declared_and_reported(dirs):
    """Gate 4: tolerance/threshold are explicit, documented, and reported
    in the output -- both at the top level and per near-threshold finding,
    with the measured distance, never silently rounded."""
    result, _ = lint(dirs["clean"])
    assert result["manifest"]["normalization_tolerance"] == pytest.approx(1e-3)
    assert result["manifest"]["duplicate_threshold"] == pytest.approx(0.999)

    mixed_result, _ = lint(dirs["defect_mixed_norm"])
    finding = mixed_result["findings"][0]
    assert finding["evidence"]["tolerance"] == pytest.approx(1e-3)
    for measured in finding["evidence"]["measured_norm_deviation"].values():
        assert isinstance(measured, float)
        assert measured > 1e-3  # actually crossed tolerance, reported exactly

    dup_result, _ = lint(dirs["defect_dup_vectors"])
    dup_finding = dup_result["findings"][0]
    assert dup_finding["evidence"]["threshold"] == pytest.approx(0.999)
    pair = dup_finding["evidence"]["pairs"][0]
    assert pair["cosine_similarity"] >= 0.999
    assert pair["cosine_similarity"] < 1.0  # near, not identical -- measured, not rounded


@pytest.mark.parametrize("name", sorted(k for k in [
    "clean", "clean_l2", "clean_npy_format",
    "defect_mixed_norm", "defect_dim_mismatch", "defect_nonfinite",
    "defect_zero_vector", "defect_dup_ids", "defect_orphaned",
    "defect_stale_ref", "defect_dup_vectors", "defect_metric_mismatch",
]))
def test_every_finding_id_resolves_in_input(name, dirs):
    """Verdict contract: every id in a finding MUST resolve in the supplied
    input. lint() itself asserts this internally (raises on failure); this
    test additionally re-checks from outside against the raw fixture spec."""
    import json
    import os

    result, _ = lint(dirs[name])

    vec_ids = set()
    with open(os.path.join(dirs[name], "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["format"] == "veclint-jsonl-v1":
        with open(os.path.join(dirs[name], "vectors.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    vec_ids.add(json.loads(line)["id"])
    else:
        with open(os.path.join(dirs[name], "vectors.ids.json"), encoding="utf-8") as f:
            vec_ids.update(json.load(f))
    map_ids = set()
    with open(os.path.join(dirs[name], "mapping.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                map_ids.add(json.loads(line)["id"])
    known = vec_ids | map_ids

    for finding in result["findings"]:
        for fid in finding["ids"]:
            assert fid in known, f"{name}: finding {finding['rule_id']} cites unresolved id {fid!r}"


def test_duplicate_vectors_no_self_pair_when_id_is_duplicated():
    """Regression: rule 8
    is defined over DISTINCT ids. If the same id string appears twice in
    vectors.jsonl (rule 5's territory) and its two rows happen to also be
    near-identical vectors, rule 8 must never emit an a==b self-pair in
    evidence.pairs -- the distinct-id invariant is enforced in the pairing
    itself, not bolted on as a duplicate_ids special case."""
    import tempfile

    import numpy as np

    import gen_corpus

    rng = np.random.RandomState(4242)
    n = 6
    ids = [f"z{i}" for i in range(n)]
    vecs = gen_corpus._unit_vectors(rng, n)

    dup_id = "z0"
    # a second row under the SAME id, near-identical to the first z0 row --
    # this is exactly the shape that used to leak a (z0, z0) self-pair.
    base = vecs[0]
    sigma = 0.02
    twin = base
    for _ in range(20):
        noisy = base + rng.normal(size=8) * sigma
        noisy = noisy / np.linalg.norm(noisy)
        cos = float(np.dot(base, noisy))
        if 0.999 <= cos < 1.0:
            twin = noisy
            break
        sigma *= 0.5

    all_ids = ids + [dup_id]
    all_vecs = [v.tolist() for v in vecs] + [twin.tolist()]
    mapping = [(i, f"{i}.txt") for i in ids]  # mapping has dup_id only once
    corpus = {f"{i}.txt": "x" for i in ids}
    manifest = gen_corpus._basic_manifest("cosine", "cosine")

    with tempfile.TemporaryDirectory() as tmp:
        gen_corpus._write_index(tmp, manifest, list(zip(all_ids, all_vecs)), mapping, corpus)
        result, exit_code = lint(tmp)

    assert exit_code == 1  # duplicate_ids must still fire
    rule_ids = {f["rule_id"] for f in result["findings"]}
    assert "duplicate_ids" in rule_ids

    dup_vec_findings = [f for f in result["findings"] if f["rule_id"] == "duplicate_vectors"]
    for finding in dup_vec_findings:
        for pair in finding["evidence"]["pairs"]:
            assert pair["a"] != pair["b"], f"rule 8 emitted a self-pair: {pair}"
