import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

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


def _json_response(payload):
    response = Mock(status_code=200)
    response.content = json.dumps(payload).encode()
    return response


def _station_info():
    return {
        "station_name": "Fredericksburg",
        "iaga_code": "FRD",
        "longitude": 282.627,
        "latitude": 38.205,
        "altitude": 69,
    }


def test_intermag_client_falls_back_from_empty_xyz_to_hdz():
    client = IntermagClient(
        observatory="FRD",
        start="2015-06-22T00:00:00",
        end="2015-06-23T00:00:00",
        elements=["X", "Y", "Z"],
        baseline_declination=-10.0,
    )
    client._request_data = Mock(
        side_effect=[
            _json_response(
                {
                    "@info": _station_info(),
                    "datetime": ["2015-06-22T00:00:00Z", "2015-06-22T00:00:01Z"],
                    "X": [None, None],
                    "Y": [None, None],
                    "Z": [30.0, 31.0],
                }
            ),
            _json_response(
                {
                    "@info": _station_info(),
                    "datetime": ["2015-06-22T00:00:00Z", "2015-06-22T00:00:01Z"],
                    "H": [100.0, 200.0],
                    "D": [0.0, 90.0],
                    "Z": [30.0, 31.0],
                }
            ),
        ]
    )

    run = client.get_data()

    np.testing.assert_allclose(
        run.hx.data_array.data,
        [100 * np.cos(np.deg2rad(-10)), 200 * np.cos(np.deg2rad(80))],
    )
    np.testing.assert_allclose(
        run.hy.data_array.data,
        [100 * np.sin(np.deg2rad(-10)), 200 * np.sin(np.deg2rad(80))],
    )
    np.testing.assert_allclose(run.hz.data_array.data, [30.0, 31.0])
    assert client._request_data.call_count == 2
    fallback_params = client._request_data.call_args_list[1].args[0]["params"]
    assert fallback_params["elements"] == "H,D,Z"
    assert fallback_params["publicationState"] == "reported"


def test_intermag_client_can_disable_hdz_fallback():
    client = IntermagClient(
        observatory="FRD",
        start="2015-06-22T00:00:00",
        end="2015-06-23T00:00:00",
        elements=["X", "Y", "Z"],
        fallback_to_hdz=False,
    )
    client._request_data = Mock(
        return_value=_json_response(
            {
                "@info": _station_info(),
                "datetime": ["2015-06-22T00:00:00Z"],
                "X": [None],
                "Y": [None],
                "Z": [30.0],
            }
        )
    )

    run = client.get_data()

    assert np.isnan(run.hx.data_array.data).all()
    assert np.isnan(run.hy.data_array.data).all()
    assert client._request_data.call_count == 1

# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])