import os
import sys

# Make the veclint package importable when pytest is run from anywhere,
# without requiring an editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import gen_corpus  # noqa: E402


_dirs_cache = {}


@pytest.fixture(scope="session", autouse=True)
def _generate_once(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("veclint_fixtures_root")
    specs = gen_corpus.generate_all(str(out_dir))
    _dirs_cache["dirs"] = {name: os.path.join(str(out_dir), name) for name in specs}
    _dirs_cache["specs"] = specs
    yield


@pytest.fixture(scope="session")
def specs():
    return _dirs_cache["specs"]


@pytest.fixture(scope="session")
def dirs():
    return _dirs_cache["dirs"]
