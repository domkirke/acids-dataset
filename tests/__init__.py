import pytest
import os, sys
from pathlib import Path
import gin

CURRENT_TEST_DIR = Path(__file__).parent
OUT_TEST_DIR = CURRENT_TEST_DIR / "outs"
os.makedirs(OUT_TEST_DIR, exist_ok=True)

sys.path.append(str((CURRENT_TEST_DIR / '..').resolve()))
gin.add_config_file_search_path(str((CURRENT_TEST_DIR / '..' / 'acids_dataset' / 'configs').resolve()))

@pytest.fixture
def test_name(request):
    return request.node.name