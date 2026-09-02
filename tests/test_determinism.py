"""Gate 5: determinism across PROCESSES, >= 3 subprocesses with differing
PYTHONHASHSEED, byte-identical stdout."""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(index_dir, hash_seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    proc = subprocess.run(
        [sys.executable, "-m", "veclint", "--index-dir", index_dir],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
    )
    return proc.stdout, proc.returncode


def test_byte_identical_stdout_across_hash_seeds_clean(dirs):
    outputs = [_run(dirs["clean"], seed) for seed in (0, 1, 2)]
    stdouts = [o for o, _ in outputs]
    codes = [c for _, c in outputs]
    assert len(set(stdouts)) == 1, "stdout differed across PYTHONHASHSEED values"
    assert len(set(codes)) == 1
    assert codes[0] == 0


def test_byte_identical_stdout_across_hash_seeds_with_findings(dirs):
    # A fixture that exercises multiple rules with dict/set-shaped internal
    # state (duplicate ids, orphaned ids) is the sharpest test of ordering
    # stability -- use defect_dup_vectors (pairs + sets) and defect_mixed_norm
    # (dict of deviations) across four different seeds including "random".
    for name in ("defect_dup_vectors", "defect_mixed_norm", "defect_dup_ids"):
        outputs = [_run(dirs[name], seed) for seed in (0, 17, 4242)]
        stdouts = [o for o, _ in outputs]
        assert len(set(stdouts)) == 1, f"{name}: stdout differed across PYTHONHASHSEED values"


def test_four_subprocesses_agree(dirs):
    outputs = [_run(dirs["clean"], seed) for seed in (0, 1, 2, 3)]
    stdouts = {o for o, _ in outputs}
    assert len(stdouts) == 1
