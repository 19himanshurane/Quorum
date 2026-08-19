import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["ARBITRATION_PROVIDER_MODE"] = "mock"

import pytest


@pytest.fixture()
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "arbitration_test.db")
