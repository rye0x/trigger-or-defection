import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tok():
    """Qwen tokenizer from the local HF cache; skip if it isn't downloaded."""
    from transformers import AutoTokenizer
    from src.data import MODEL_ID
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not in the local HF cache")
