"""Gate 6: the linter can go red. Positive control -- mutate a rule and
confirm the gate that used to catch a planted defect now fails to."""
from unittest import mock

from veclint import cli, rules
from veclint.cli import lint


def test_baseline_detects_dimension_mismatch(dirs):
    result, exit_code = lint(dirs["defect_dim_mismatch"])
    assert exit_code == 1
    assert any(f["rule_id"] == "dimension_mismatch" for f in result["findings"])


def test_mutated_rule_goes_red(dirs):
    """Disable rule_dimension_mismatch (simulate a regression that breaks
    it) and confirm the previously-caught defect now slips through silently
    -- proving the test suite actually depends on the rule's code, not a
    vacuous always-pass."""
    mutated_rules = [r for r in rules.ALL_RULES if r is not rules.rule_dimension_mismatch]
    mutated_rules.append(lambda index: [])  # no-op stand-in

    with mock.patch.object(cli, "ALL_RULES", mutated_rules):
        result, exit_code = lint(dirs["defect_dim_mismatch"])

    dim_findings = [f for f in result["findings"] if f["rule_id"] == "dimension_mismatch"]
    assert dim_findings == [], "mutated rule should have gone silent, but still fired"
    assert exit_code == 0, "with the mutated rule, this fixture must now read clean -- proving the gate is live"


def test_mutated_tolerance_changes_mixed_norm_verdict(dirs):
    """A second, independent mutation: loosen normalization_tolerance to
    something absurd and confirm the previously-firing rule 1 finding on
    defect_mixed_norm disappears."""
    import json
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(dirs["defect_mixed_norm"], tmp, dirs_exist_ok=True)
        manifest_path = os.path.join(tmp, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["normalization_tolerance"] = 10.0  # absurdly loose -- mutation
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        result, exit_code = lint(tmp)

    norm_findings = [f for f in result["findings"] if f["rule_id"] == "mixed_normalization"]
    assert norm_findings == [], "loosened tolerance should silence the previously-firing rule"
    assert exit_code == 0
