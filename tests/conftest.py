import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))


@pytest.fixture(autouse=True, scope="session")
def private_grader_cache(tmp_path_factory):
    """Point the grader's verdict cache at a temporary copy, so that tests never
    modify the committed cache."""
    from tokenguard.reasoning import datasets

    copy = tmp_path_factory.mktemp("grader") / "grader_cache.json"
    shutil.copy(datasets._GC_PATH, copy)
    datasets._GC_PATH, datasets._GC = copy, None
    yield
