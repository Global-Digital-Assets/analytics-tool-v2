import importlib
import os


def test_env_flags_adaptive_tiers():
    # Ensure the flag is read from env variable
    os.environ["ENABLE_ADAPTIVE_TIERS"] = "true"
    mta = importlib.reload(importlib.import_module("multi_token_analyzer"))
    assert mta.ENABLE_ADAPTIVE_TIERS is True
    os.environ.pop("ENABLE_ADAPTIVE_TIERS")
