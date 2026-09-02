"""Coverage-is-a-declared-table behavior: an index format outside
SUPPORTED_FORMATS is reported `unsupported`, never guess-parsed. Malformed
input (missing files, bad JSON) is exit code 2, distinct from findings."""
import json
import os

from veclint.cli import lint


def test_unsupported_format_reported_not_guess_parsed(dirs):
    result, exit_code = lint(dirs["unsupported_format"])
    assert exit_code == 2
    assert result["status"] == "unsupported"
    assert result["format"] == "veclint-csv-v1"
    assert result["findings"] == []
    assert "veclint-jsonl-v1" in result["supported_formats"]
    assert "veclint-npy-v1" in result["supported_formats"]


def test_missing_manifest_is_malformed_exit_2(tmp_path):
    empty_dir = tmp_path / "empty_index"
    empty_dir.mkdir()
    result, exit_code = lint(str(empty_dir))
    assert exit_code == 2
    assert result["status"] == "malformed"


def test_missing_vectors_file_is_malformed_exit_2(tmp_path, dirs):
    import shutil

    broken = tmp_path / "broken_index"
    shutil.copytree(dirs["clean"], broken)
    os.remove(broken / "vectors.jsonl")
    result, exit_code = lint(str(broken))
    assert exit_code == 2
    assert result["status"] == "malformed"


def test_bad_json_in_manifest_is_malformed_exit_2(tmp_path):
    d = tmp_path / "bad_manifest"
    d.mkdir()
    (d / "manifest.json").write_text("{not valid json", encoding="utf-8")
    result, exit_code = lint(str(d))
    assert exit_code == 2
    assert result["status"] == "malformed"


def test_manifest_missing_required_key_is_malformed(tmp_path):
    d = tmp_path / "incomplete_manifest"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"format": "veclint-jsonl-v1", "dimension": 4}), encoding="utf-8"
    )
    result, exit_code = lint(str(d))
    assert exit_code == 2
    assert result["status"] == "malformed"
