"""Whole-corpus honesty metrics, computed and reported SEPARATELY, never
blended: detection_rate, false_flag_rate, unsupported_rate."""
from veclint.cli import lint

DEFECT_FIXTURES = [
    "defect_mixed_norm",
    "defect_dim_mismatch",
    "defect_nonfinite",
    "defect_zero_vector",
    "defect_dup_ids",
    "defect_orphaned",
    "defect_stale_ref",
    "defect_dup_vectors",
    "defect_metric_mismatch",
]
CLEAN_FIXTURES = ["clean", "clean_l2", "clean_npy_format"]
UNSUPPORTED_FIXTURES = ["unsupported_format"]


def compute_corpus_metrics(specs, dirs):
    detected = 0
    for name in DEFECT_FIXTURES:
        spec = specs[name]
        result, _ = lint(dirs[name])
        got_rule_ids = {f["rule_id"] for f in result["findings"]}
        expected_rule_ids = set(spec["expected_rule_ids"])
        ids_ok = all(
            sorted(next(f["ids"] for f in result["findings"] if f["rule_id"] == rid)) == sorted(expected_ids)
            for rid, expected_ids in spec["expected_ids"].items()
        )
        if got_rule_ids == expected_rule_ids and ids_ok:
            detected += 1
    detection_rate = detected / len(DEFECT_FIXTURES)

    false_flags = 0
    for name in CLEAN_FIXTURES:
        result, _ = lint(dirs[name])
        if result["findings"]:
            false_flags += 1
    false_flag_rate = false_flags / len(CLEAN_FIXTURES)

    correctly_unsupported = 0
    for name in UNSUPPORTED_FIXTURES:
        result, exit_code = lint(dirs[name])
        if result["status"] == "unsupported" and exit_code == 2:
            correctly_unsupported += 1
    unsupported_rate = correctly_unsupported / len(UNSUPPORTED_FIXTURES)

    # honesty check the other direction: supported fixtures must never be
    # misclassified as unsupported (not folded into unsupported_rate --
    # never blend two-sided results into one number).
    supported_misclassified = 0
    for name in DEFECT_FIXTURES + CLEAN_FIXTURES:
        result, _ = lint(dirs[name])
        if result.get("status") == "unsupported":
            supported_misclassified += 1

    return {
        "detection_rate": detection_rate,
        "false_flag_rate": false_flag_rate,
        "unsupported_rate": unsupported_rate,
        "supported_misclassified_as_unsupported": supported_misclassified,
    }


def test_detection_rate_is_1(specs, dirs):
    metrics = compute_corpus_metrics(specs, dirs)
    assert metrics["detection_rate"] == 1.0, metrics


def test_false_flag_rate_is_0(specs, dirs):
    metrics = compute_corpus_metrics(specs, dirs)
    assert metrics["false_flag_rate"] == 0.0, metrics


def test_unsupported_rate_is_1_and_no_supported_misclassified(specs, dirs):
    metrics = compute_corpus_metrics(specs, dirs)
    assert metrics["unsupported_rate"] == 1.0, metrics
    assert metrics["supported_misclassified_as_unsupported"] == 0, metrics


def test_print_corpus_metrics_summary(specs, dirs, capsys):
    """Not an assertion -- prints the honest summary line for the report."""
    metrics = compute_corpus_metrics(specs, dirs)
    print(
        f"CORPUS_METRICS detection_rate={metrics['detection_rate']:.4f} "
        f"false_flag_rate={metrics['false_flag_rate']:.4f} "
        f"unsupported_rate={metrics['unsupported_rate']:.4f}"
    )
