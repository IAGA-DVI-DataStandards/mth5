# -*- coding: utf-8 -*-
"""
Combined pytest suite for GeomagClient and USGSGeomag classes with comprehensive
testing including fixtures, subtests, and mocking for external dependencies.

Created by combining test_geomag_client.py and test_geomag.py with modern pytest patterns.

@author: pytest conversion
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from mt_timeseries import ChannelTS, RunTS

from mth5.clients.geomag import GeomagClient, USGSGeomag


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def geomag_client():
    """Create a fresh GeomagClient instance for each test."""
    return GeomagClient()


@pytest.fixture
def configured_geomag_client():
    """Create a configured GeomagClient for data retrieval tests."""
    client = GeomagClient()
    client.observatory = "frn"
    client.sampling_period = 1
    client.start = "2020-01-01T00:00:00"
    client.end = "2020-01-01T12:00:00"
    client.elements = ["x", "y"]
    return client


@pytest.fixture
def usgs_geomag_client():
    """Create a fresh USGSGeomag instance for each test."""
    return USGSGeomag()


@pytest.fixture
def sample_request_df():
    """Create a sample request DataFrame for USGSGeomag tests."""
    return pd.DataFrame(
        {
            "observatory": ["frn", "frn", "ott", "ott"],
            "type": ["adjusted"] * 4,
            "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
            "sampling_period": [1, 1, 1, 1],
            "start": [
                "2022-01-01T00:00:00",
                "2022-01-03T00:00:00",
                "2022-01-01T00:00:00",
                "2022-01-03T00:00:00",
            ],
            "end": [
                "2022-01-02T00:00:00",
                "2022-01-04T00:00:00",
                "2022-01-02T00:00:00",
                "2022-01-04T00:00:00",
            ],
        }
    )


@pytest.fixture
def mock_geomag_response():
    """Create a mock response from USGS geomag web service."""
    # Create time stamps as string list for JSON compatibility
    time_stamps = (
        pd.date_range(
            "2020-01-01T00:00:00",
            "2020-01-01T12:00:00",
            freq="1s",  # Use lowercase 's' instead of deprecated 'S'
        )
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        .tolist()
    )

    return {
        "metadata": {
            "intermagnet": {
                "imo": {
                    "name": "Fresno",
                    "iaga_code": "FRN",
                    "coordinates": [-119.718, 37.091, 331.0],
                }
            },
            "generated": "2023-03-20T23:02:17+00:00",
        },
        "times": time_stamps,
        "values": [
            {
                "metadata": {"element": "X"},
                "values": np.random.normal(22627, 0.1, len(time_stamps)).tolist(),
            },
            {
                "metadata": {"element": "Y"},
                "values": np.random.normal(5111, 0.1, len(time_stamps)).tolist(),
            },
        ],
    }


@pytest.fixture
def mock_http_response(mock_geomag_response):
    """Create a mock HTTP response object."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = json.dumps(mock_geomag_response).encode()
    return mock_response


# =============================================================================
# GeomagClient Property Tests
# =============================================================================


class TestGeomagClientProperties:
    """Test GeomagClient property validation and functionality."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [("frn", "FRN"), ("FRN", "FRN"), ("bou", "BOU"), ("tst", "TST")],
    )
    def test_observatory_valid_setting(self, geomag_client, input_val, expected):
        """Test setting valid observatory codes."""
        geomag_client.observatory = input_val
        assert geomag_client.observatory == expected

    @pytest.mark.parametrize(
        "invalid_val,expected_exc,description",
        [
            ("ten", ValueError, "invalid observatory code"),
            (10, TypeError, "non-string input"),
            (None, TypeError, "None input"),
            ([], TypeError, "list input"),
        ],
    )
    def test_observatory_invalid_values(
        self, geomag_client, invalid_val, expected_exc, description
    ):
        """Test observatory validation with invalid values."""
        with pytest.raises(expected_exc):
            geomag_client.observatory = invalid_val

    def test_elements_single_string(self, geomag_client):
        """Test setting elements from single string."""
        geomag_client.elements = "x"
        assert geomag_client.elements == ["X"]

    def test_elements_comma_separated_string(self, geomag_client):
        """Test setting elements from comma-separated string."""
        geomag_client.elements = "x,y"
        assert geomag_client.elements == ["X", "Y"]

    def test_elements_list_input(self, geomag_client):
        """Test setting elements from list."""
        geomag_client.elements = ["x", "y", "z"]
        assert geomag_client.elements == ["X", "Y", "Z"]

    @pytest.mark.parametrize(
        "invalid_val,expected_exc,description",
        [
            ("v", ValueError, "invalid single element"),
            (["x", "v"], ValueError, "invalid element in list"),
            (10, TypeError, "non-string/list input"),
            (["x", 10], TypeError, "non-string item in list"),
        ],
    )
    def test_elements_invalid_values(
        self, geomag_client, invalid_val, expected_exc, description
    ):
        """Test elements validation with invalid values."""
        with pytest.raises(expected_exc):
            geomag_client.elements = invalid_val

    @pytest.mark.parametrize(
        "period,expected",
        [(1, 1), (60, 60), (3600, 3600), ("1", 1), ("60", 60), ("3600", 3600)],
    )
    def test_sampling_period_valid_values(self, geomag_client, period, expected):
        """Test setting valid sampling periods."""
        geomag_client.sampling_period = period
        assert geomag_client.sampling_period == expected

    @pytest.mark.parametrize(
        "invalid_val,expected_exc,description",
        [
            ("p", ValueError, "non-numeric string"),
            ([1], TypeError, "list input"),
            (10, ValueError, "invalid numeric value"),
            (1.5, ValueError, "non-integer float"),
            (None, TypeError, "None input"),
        ],
    )
    def test_sampling_period_invalid_values(
        self, geomag_client, invalid_val, expected_exc, description
    ):
        """Test sampling period validation with invalid values."""
        with pytest.raises(expected_exc):
            geomag_client.sampling_period = invalid_val

    def test_time_properties(self, geomag_client):
        """Test start and end time property setting."""
        test_time = "2020-01-01T00:00:00+00:00"
        expected_time = "2020-01-01T00:00:00Z"

        geomag_client.start = test_time
        assert geomag_client.start == expected_time

        geomag_client.end = test_time
        assert geomag_client.end == expected_time

    def test_user_agent_property(self, geomag_client):
        """Test user agent string generation."""
        user_agent = geomag_client.user_agent
        assert "MTH5" in user_agent
        assert "Python" in user_agent


# =============================================================================
# GeomagClient Functionality Tests
# =============================================================================


class TestGeomagClientFunctionality:
    """Test GeomagClient core functionality."""

    def test_get_chunks(self, geomag_client):
        """Test chunk calculation for large time ranges."""
        geomag_client.start = "2021-04-05T00:00:00+00:00"
        geomag_client.end = "2021-04-16T00:00:00+00:00"

        expected_chunks = [
            ("2021-04-05T00:00:00Z", "2021-04-05T20:00:00Z"),
            ("2021-04-05T20:00:00Z", "2021-04-06T16:00:00Z"),
            ("2021-04-06T16:00:00Z", "2021-04-07T12:00:00Z"),
            ("2021-04-07T12:00:00Z", "2021-04-08T08:00:00Z"),
            ("2021-04-08T08:00:00Z", "2021-04-09T04:00:00Z"),
            ("2021-04-09T04:00:00Z", "2021-04-10T00:00:00Z"),
            ("2021-04-10T00:00:00Z", "2021-04-10T20:00:00Z"),
            ("2021-04-10T20:00:00Z", "2021-04-11T16:00:00Z"),
            ("2021-04-11T16:00:00Z", "2021-04-12T12:00:00Z"),
            ("2021-04-12T12:00:00Z", "2021-04-13T08:00:00Z"),
            ("2021-04-13T08:00:00Z", "2021-04-14T04:00:00Z"),
            ("2021-04-14T04:00:00Z", "2021-04-15T00:00:00Z"),
            ("2021-04-15T00:00:00Z", "2021-04-15T20:00:00Z"),
            ("2021-04-15T20:00:00Z", "2021-04-16T00:00:00Z"),
        ]

        assert geomag_client.get_chunks() == expected_chunks

    def test_get_request_params(self, configured_geomag_client):
        """Test request parameter generation."""
        start = "2020-01-01T00:00:00+00:00"
        end = "2020-01-02T12:00:00+00:00"

        expected_params = {
            "id": "FRN",
            "type": "adjusted",
            "elements": "X,Y",
            "sampling_period": 1,
            "format": "json",
            "starttime": start,
            "endtime": end,
        }

        actual_params = configured_geomag_client._get_request_params(start, end)
        assert actual_params == expected_params

    def test_get_request_dictionary(self, configured_geomag_client):
        """Test request dictionary generation."""
        start = "2020-01-01T00:00:00+00:00"
        end = "2020-01-02T12:00:00+00:00"

        request_dict = configured_geomag_client._get_request_dictionary(start, end)

        assert "url" in request_dict
        assert "headers" in request_dict
        assert "params" in request_dict
        assert "timeout" in request_dict
        assert (
            request_dict["headers"]["User-Agent"] == configured_geomag_client.user_agent
        )


# =============================================================================
# GeomagClient Data Retrieval Tests (Mocked)
# =============================================================================


class TestGeomagClientDataRetrieval:
    """Test GeomagClient data retrieval with mocked HTTP requests."""

    @patch("mth5.clients.geomag.requests.get")
    def test_get_data_successful_request(
        self, mock_get, configured_geomag_client, mock_http_response
    ):
        """Test successful data retrieval with mocked response."""
        mock_get.return_value = mock_http_response

        # Test data retrieval
        result = configured_geomag_client.get_data()

        # Verify request was made
        mock_get.assert_called_once()

        # Verify result structure
        assert isinstance(result, RunTS)
        assert hasattr(result, "hx")
        assert hasattr(result, "hy")
        assert isinstance(result.hx, ChannelTS)
        assert isinstance(result.hy, ChannelTS)

    @patch("mth5.clients.geomag.requests.get")
    def test_get_data_server_error(self, mock_get, configured_geomag_client):
        """Test handling of server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(IOError, match="Could not connect to server"):
            configured_geomag_client.get_data()

    @patch("mth5.clients.geomag.requests.get")
    def test_get_data_metadata_validation(
        self,
        mock_get,
        configured_geomag_client,
        mock_geomag_response,
        mock_http_response,
    ):
        """Test metadata validation in retrieved data."""
        mock_get.return_value = mock_http_response

        result = configured_geomag_client.get_data()

        # Test survey metadata
        assert result.survey_metadata.id == "USGS-GEOMAG"

        # Test station metadata
        assert result.station_metadata.id == "Fresno"
        assert result.station_metadata.fdsn.id == "FRN"
        assert result.station_metadata.location.latitude == 37.091
        assert result.station_metadata.location.longitude == -119.718
        assert result.station_metadata.location.elevation == 331.0

    @patch("mth5.clients.geomag.requests.get")
    def test_get_data_channel_metadata(
        self, mock_get, configured_geomag_client, mock_http_response
    ):
        """Test channel metadata in retrieved data."""
        mock_get.return_value = mock_http_response

        result = configured_geomag_client.get_data()

        # Test hx channel metadata
        hx_metadata = result.hx.channel_metadata
        assert hx_metadata.component == "hx"
        assert hx_metadata.measurement_azimuth == 0.0
        assert hx_metadata.units == "nanoTesla"  # Note: actual value from geomag client
        assert hx_metadata.sample_rate == 1.0

        # Test hy channel metadata
        hy_metadata = result.hy.channel_metadata
        assert hy_metadata.component == "hy"
        assert hy_metadata.measurement_azimuth == 90.0
        assert hy_metadata.units == "nanoTesla"  # Note: actual value from geomag client
        assert hy_metadata.sample_rate == 1.0

    @patch("mth5.clients.geomag.requests.get")
    def test_to_station_metadata(
        self, mock_get, configured_geomag_client, mock_geomag_response
    ):
        """Test station metadata conversion."""
        metadata = mock_geomag_response["metadata"]
        station_metadata = configured_geomag_client._to_station_metadata(metadata)

        assert station_metadata.id == "Fresno"
        assert station_metadata.fdsn.id == "FRN"
        assert station_metadata.location.longitude == -119.718
        assert station_metadata.location.latitude == 37.091
        assert station_metadata.location.elevation == 331.0


# =============================================================================
# USGSGeomag Tests
# =============================================================================


class TestUSGSGeomagProperties:
    """Test USGSGeomag initialization and properties."""

    def test_initialization_defaults(self, usgs_geomag_client):
        """Test default initialization values."""
        assert usgs_geomag_client.save_path == Path().cwd()
        assert usgs_geomag_client.mth5_filename is None
        assert usgs_geomag_client.interact is False
        assert usgs_geomag_client.h5_compression == "gzip"
        assert usgs_geomag_client.h5_compression_opts == 4
        assert usgs_geomag_client.mth5_version == "0.2.0"

    def test_initialization_with_kwargs(self):
        """Test initialization with custom parameters."""
        client = USGSGeomag(
            save_path="/custom/path",
            interact=True,
            h5_compression="lzf",
            mth5_version="0.1.0",
        )

        assert str(client.save_path) == "/custom/path"
        assert client.interact is True
        assert client.h5_compression == "lzf"
        assert client.mth5_version == "0.1.0"

    def test_h5_kwargs_property(self, usgs_geomag_client):
        """Test HDF5 parameters property."""
        h5_kwargs = usgs_geomag_client.h5_kwargs

        expected_keys = [
            "file_version",
            "compression",
            "compression_opts",
            "shuffle",
            "fletcher32",
            "data_level",
        ]

        for key in expected_keys:
            assert key in h5_kwargs

        assert h5_kwargs["file_version"] == "0.2.0"
        assert h5_kwargs["compression"] == "gzip"
        assert h5_kwargs["compression_opts"] == 4

    def test_request_columns(self, usgs_geomag_client):
        """Test required request columns."""
        expected_columns = [
            "observatory",
            "type",
            "elements",
            "sampling_period",
            "start",
            "end",
        ]

        assert usgs_geomag_client.request_columns == expected_columns


# =============================================================================
# USGSGeomag Request DataFrame Tests
# =============================================================================


class TestUSGSGeomagRequestValidation:
    """Test USGSGeomag request DataFrame validation."""

    def test_validate_request_df_valid_input(
        self, usgs_geomag_client, sample_request_df
    ):
        """Test validation with valid DataFrame."""
        validated_df = usgs_geomag_client.validate_request_df(sample_request_df)

        assert isinstance(validated_df, pd.DataFrame)
        assert "run" in validated_df.columns
        assert len(validated_df) == 4

    def test_validate_request_df_invalid_file(self, usgs_geomag_client):
        """Test validation with non-existent file."""
        with pytest.raises(IOError, match="does not exist"):
            usgs_geomag_client.validate_request_df("nonexistent_file.csv")

    def test_validate_request_df_invalid_type(self, usgs_geomag_client):
        """Test validation with invalid input type."""
        with pytest.raises(TypeError, match="must be a pandas.DataFrame"):
            usgs_geomag_client.validate_request_df(10)

    def test_validate_request_df_missing_columns(self, usgs_geomag_client):
        """Test validation with missing required columns."""
        invalid_df = pd.DataFrame({"a": [10]})

        with pytest.raises(ValueError, match="Request must have columns"):
            usgs_geomag_client.validate_request_df(invalid_df)

    def test_add_run_id_basic(self, usgs_geomag_client, sample_request_df):
        """Test adding run IDs to request DataFrame."""
        result_df = usgs_geomag_client.add_run_id(sample_request_df)

        expected_runs = ["sp1_001", "sp1_002", "sp1_001", "sp1_002"]
        assert result_df.run.tolist() == expected_runs

    def test_add_run_id_different_sampling_periods(self, usgs_geomag_client):
        """Test adding run IDs with different sampling periods."""
        request_df = pd.DataFrame(
            {
                "observatory": ["frn", "frn", "ott", "ott"],
                "type": ["adjusted"] * 4,
                "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
                "sampling_period": [1, 60, 1, 60],
                "start": [
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                ],
                "end": [
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                ],
            }
        )

        result_df = usgs_geomag_client.add_run_id(request_df)
        expected_runs = ["sp1_001", "sp60_001", "sp1_001", "sp60_001"]
        assert result_df.run.tolist() == expected_runs

    def test_make_filename(self, usgs_geomag_client, sample_request_df):
        """Test filename generation from request DataFrame."""
        filename = usgs_geomag_client._make_filename(Path(), sample_request_df)

        expected_filename = Path().joinpath("usgs_geomag_frn_ott_xy.h5")
        assert filename == expected_filename

    def test_make_filename_single_observatory(self, usgs_geomag_client):
        """Test filename generation with single observatory."""
        single_obs_df = pd.DataFrame(
            {"observatory": ["frn", "frn"], "elements": [["x", "y", "z"], ["h"]]}
        )

        filename = usgs_geomag_client._make_filename(Path(), single_obs_df)
        expected_filename = Path().joinpath("usgs_geomag_frn_xyzh.h5")
        assert filename == expected_filename


# =============================================================================
# USGSGeomag MTH5 Creation Tests (Mocked)
# =============================================================================


class TestUSGSGeomagMTH5Creation:
    """Test USGSGeomag MTH5 file creation with mocked dependencies."""

    @patch("mth5.clients.geomag.MTH5")
    @patch("mth5.clients.geomag.GeomagClient")
    def test_make_mth5_from_geomag_mocked(
        self,
        mock_geomag_client_class,
        mock_mth5_class,
        usgs_geomag_client,
        sample_request_df,
    ):
        """Test MTH5 creation with fully mocked dependencies."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance
        mock_mth5_instance.filename = Path("test_file.h5")

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup mock GeomagClient
        mock_client_instance = MagicMock()
        mock_geomag_client_class.return_value = mock_client_instance

        mock_run_data = MagicMock()
        mock_run_data.station_metadata.id = "TEST_STATION"
        mock_run_data.run_metadata.id = "sp1_001"
        mock_client_instance.get_data.return_value = mock_run_data

        # Test the method
        usgs_geomag_client.interact = False
        result = usgs_geomag_client.make_mth5_from_geomag(sample_request_df)

        # Verify result
        assert result == Path("test_file.h5")

        # Verify MTH5 was created with correct parameters
        mock_mth5_class.assert_called_once()
        mock_mth5_instance.open_mth5.assert_called()
        mock_mth5_instance.add_survey.assert_called_with("USGS-GEOMAG")

    @patch("mth5.clients.geomag.MTH5")
    @patch("mth5.clients.geomag.GeomagClient")
    def test_make_mth5_from_geomag_interact_mode(
        self,
        mock_geomag_client_class,
        mock_mth5_class,
        usgs_geomag_client,
        sample_request_df,
    ):
        """Test MTH5 creation in interactive mode."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance
        mock_mth5_instance.filename = Path("test_file.h5")

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup mock GeomagClient
        mock_client_instance = MagicMock()
        mock_geomag_client_class.return_value = mock_client_instance

        mock_run_data = MagicMock()
        mock_run_data.station_metadata.id = "TEST_STATION"
        mock_run_data.run_metadata.id = "sp1_001"
        mock_client_instance.get_data.return_value = mock_run_data

        # Test interactive mode
        usgs_geomag_client.interact = True
        result = usgs_geomag_client.make_mth5_from_geomag(sample_request_df)

        # Verify interactive mode returns MTH5 instance
        assert result == mock_mth5_instance

    @patch("mth5.clients.geomag.MTH5")
    def test_make_mth5_invalid_version(
        self, mock_mth5_class, usgs_geomag_client, sample_request_df
    ):
        """Test MTH5 creation with invalid version."""
        usgs_geomag_client.mth5_version = "0.3.0"  # Invalid version

        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        with pytest.raises(ValueError, match="MTH5 version must be"):
            usgs_geomag_client.make_mth5_from_geomag(sample_request_df)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_geomag_client_none_time_values(self, geomag_client):
        """Test handling of None time values."""
        geomag_client.start = None
        geomag_client.end = None

        # When both start and end are None, get_chunks should return None
        # since the property access will fail, we expect this behavior
        # In a real scenario, the user would set valid times before calling get_chunks
        with pytest.raises(AttributeError):
            # This should raise AttributeError because start property tries to access None._start.iso_no_tz
            chunks = geomag_client.get_chunks()

    def test_geomag_client_longitude_conversion(self, geomag_client):
        """Test longitude conversion for coordinates > 180."""
        metadata = {
            "intermagnet": {
                "imo": {
                    "name": "TestStation",  # No spaces to avoid pydantic validation error
                    "iaga_code": "TST",
                    "coordinates": [300.0, 45.0, 100.0],  # Longitude > 180
                }
            },
            "generated": "2023-01-01T00:00:00+00:00",
        }

        station_metadata = geomag_client._to_station_metadata(metadata)
        assert station_metadata.location.longitude == -60.0  # 300 - 360

    def test_usgs_geomag_empty_request_df(self, usgs_geomag_client):
        """Test handling of empty request DataFrame."""
        empty_df = pd.DataFrame(columns=usgs_geomag_client.request_columns)

        # Empty DataFrame should still be valid structurally,
        # but might fail at MTH5 creation stage
        validated_df = usgs_geomag_client.validate_request_df(empty_df)
        assert len(validated_df) == 0

    def test_geomag_client_invalid_elements_mixed_case(self, geomag_client):
        """Test elements validation with mixed case input."""
        geomag_client.elements = ["X", "y", "Z"]
        assert geomag_client.elements == ["X", "Y", "Z"]

    def test_geomag_client_elements_whitespace_handling(self, geomag_client):
        """Test elements handling with whitespace in comma-separated string."""
        geomag_client.elements = " x , y , z "
        assert geomag_client.elements == ["X", "Y", "Z"]


# =============================================================================
# Integration-Style Tests (Mocked)
# =============================================================================


class TestIntegrationMocked:
    """Integration-style tests with comprehensive mocking."""

    @patch("mth5.clients.geomag.requests.get")
    def test_full_workflow_single_request(self, mock_get, mock_http_response):
        """Test complete workflow for single data request."""
        mock_get.return_value = mock_http_response

        # Create and configure client
        client = GeomagClient(
            observatory="frn",
            elements=["x", "y"],
            sampling_period=1,
            start="2020-01-01T00:00:00",
            end="2020-01-01T02:00:00",
        )

        # Get data
        result = client.get_data(run_id="test_001")

        # Verify complete result structure
        assert isinstance(result, RunTS)
        assert result.run_metadata.id == "test_001"
        assert len(result.dataset.data_vars) == 2  # Should have 2 channels (hx, hy)
        assert hasattr(result, "hx")
        assert hasattr(result, "hy")

    @patch("mth5.clients.geomag.MTH5")
    @patch("mth5.clients.geomag.GeomagClient")
    def test_usgs_geomag_multiple_observatories(
        self, mock_geomag_client_class, mock_mth5_class
    ):
        """Test USGSGeomag with multiple observatories."""
        # Setup multi-observatory request
        request_df = pd.DataFrame(
            {
                "observatory": ["frn", "bou", "tst"],
                "type": ["adjusted"] * 3,
                "elements": [["x", "y"]] * 3,
                "sampling_period": [1] * 3,
                "start": ["2022-01-01T00:00:00"] * 3,
                "end": ["2022-01-01T12:00:00"] * 3,
            }
        )

        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance
        mock_mth5_instance.filename = Path("multi_obs.h5")

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_client_instance = MagicMock()
        mock_geomag_client_class.return_value = mock_client_instance

        mock_run_data = MagicMock()
        mock_run_data.station_metadata.id = "TEST_STATION"
        mock_run_data.run_metadata.id = "sp1_001"
        mock_client_instance.get_data.return_value = mock_run_data

        # Test USGSGeomag
        client = USGSGeomag(interact=False)
        result = client.make_mth5_from_geomag(request_df)

        # Verify multiple stations were processed
        assert mock_geomag_client_class.call_count == 3
        assert mock_survey_group.stations_group.add_station.call_count == 3


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_large_element_list_performance(self, geomag_client):
        """Test performance with large element lists."""
        # Test with maximum elements
        all_elements = [
            "D",
            "DIST",
            "DST",
            "E",
            "E-E",
            "E-N",
            "F",
            "G",
            "H",
            "SQ",
            "SV",
            "UK1",
            "UK2",
            "UK3",
            "UK4",
            "X",
            "Y",
            "Z",
        ]

        geomag_client.elements = all_elements
        assert len(geomag_client.elements) == len(all_elements)
        assert geomag_client.elements == all_elements

    def test_chunk_calculation_large_timespan(self, geomag_client):
        """Test chunk calculation for very large time spans."""
        geomag_client.start = "2020-01-01T00:00:00"
        geomag_client.end = "2020-12-31T23:59:59"  # Full year

        chunks = geomag_client.get_chunks()
        assert isinstance(chunks, list)
        assert len(chunks) > 50  # Should be many chunks for a full year

    def test_request_df_large_dataset(self, usgs_geomag_client):
        """Test handling of large request DataFrames."""
        # Create large request DataFrame
        n_requests = 100
        large_request_df = pd.DataFrame(
            {
                "observatory": ["frn"] * n_requests,
                "type": ["adjusted"] * n_requests,
                "elements": [["x", "y"]] * n_requests,
                "sampling_period": [1] * n_requests,
                "start": pd.date_range("2020-01-01", periods=n_requests, freq="D"),
                "end": pd.date_range("2020-01-02", periods=n_requests, freq="D"),
            }
        )

        validated_df = usgs_geomag_client.validate_request_df(large_request_df)
        assert len(validated_df) == n_requests
        assert "run" in validated_df.columns


# =============================================================================
# Compatibility Tests
# =============================================================================


class TestCompatibility:
    """Test compatibility with different data types and formats."""

    def test_geomag_client_kwargs_initialization(self):
        """Test GeomagClient initialization with various kwargs."""
        client = GeomagClient(
            observatory="BOU",
            elements=["X", "Y", "Z"],
            sampling_period=60,
            start="2020-01-01",
            end="2020-01-02",
            type="definitive",
        )

        assert client.observatory == "BOU"
        assert client.elements == ["X", "Y", "Z"]
        assert client.sampling_period == 60
        assert client.type == "definitive"

    def test_usgs_geomag_version_compatibility(self):
        """Test USGSGeomag compatibility with different MTH5 versions."""
        # Test 0.1.0 compatibility
        client_010 = USGSGeomag(mth5_version="0.1.0")
        assert client_010.mth5_version == "0.1.0"

        # Test 0.2.0 compatibility
        client_020 = USGSGeomag(mth5_version="0.2.0")
        assert client_020.mth5_version == "0.2.0"

    def test_pandas_compatibility(self, usgs_geomag_client):
        """Test compatibility with different pandas DataFrame formats."""
        # Test with string dates
        string_date_df = pd.DataFrame(
            {
                "observatory": ["frn"],
                "type": ["adjusted"],
                "elements": [["x", "y"]],
                "sampling_period": [1],
                "start": ["2022-01-01T00:00:00"],
                "end": ["2022-01-01T12:00:00"],
            }
        )

        validated_df = usgs_geomag_client.validate_request_df(string_date_df)
        assert isinstance(validated_df, pd.DataFrame)

        # Test with datetime objects
        datetime_df = pd.DataFrame(
            {
                "observatory": ["frn"],
                "type": ["adjusted"],
                "elements": [["x", "y"]],
                "sampling_period": [1],
                "start": [pd.Timestamp("2022-01-01T00:00:00")],
                "end": [pd.Timestamp("2022-01-01T12:00:00")],
            }
        )

        validated_df2 = usgs_geomag_client.validate_request_df(datetime_df)
        assert isinstance(validated_df2, pd.DataFrame)


# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])
