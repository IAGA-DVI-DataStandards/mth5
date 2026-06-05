import pytest
from pathlib import Path

from mth5.clients.intermag import Intermag, IntermagClient


# ---------------------------------------------------------------------
# Tests for IntermagClient
# ---------------------------------------------------------------------

def test_intermag_client_instantiates():
    """IntermagClient should instantiate without errors."""
    client = IntermagClient()
    assert isinstance(client, IntermagClient)


# ---------------------------------------------------------------------
# Tests for Intermag
# ---------------------------------------------------------------------

def test_intermag_initialization_defaults():
    """Intermag should initialize with correct default attributes."""
    im = Intermag()

    # save_path defaults to current working directory
    assert isinstance(im.save_path, Path)
    assert im.save_path == Path().cwd()

    # mth5_filename default
    assert im.mth5_filename is None

    # interact default
    assert im.interact is False

    # request_columns default
    expected_cols = [
        "observatory",
        "type",
        "elements",
        "sampling_period",
        "start",
        "end",
    ]
    assert im.request_columns == expected_cols


def test_intermag_allows_kwargs_without_failure():
    """Intermag accepts **kwargs even though they are unused."""
    im = Intermag(foo=123, bar="xyz")
    assert isinstance(im, Intermag)
    
# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])