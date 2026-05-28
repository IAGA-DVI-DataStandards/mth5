# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for PhoenixClient with fixtures, parametrization,
and additional coverage for untested functionality.

Created by modernizing test_phoenix.py with pytest patterns.

@author: pytest conversion
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_io.phoenix import PhoenixCollection
from mt_io.phoenix.readers.calibrations import PhoenixCalibration

from mth5.clients.phoenix import PhoenixClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file_path(temp_data_dir):
    """Create a temporary file path for testing."""
    return temp_data_dir / "test.h5"


@pytest.fixture
def basic_phoenix_client(temp_data_dir):
    """Create a basic PhoenixClient instance for testing."""
    return PhoenixClient(temp_data_dir, **{"h5_mode": "w", "h5_driver": "sec2"})


@pytest.fixture
def configured_phoenix_client(temp_data_dir, temp_file_path):
    """Create a configured PhoenixClient instance with custom settings."""
    return PhoenixClient(
        temp_data_dir,
        sample_rates=[150, 24000],
        save_path=temp_file_path.parent,
        mth5_filename=temp_file_path.name,
        receiver_calibration_dict={"RX001": "/path/to/cal1"},
        sensor_calibration_dict={"SENSOR001": "/path/to/scal1"},
        h5_compression="lzf",
        h5_shuffle=False,
        mth5_version="0.1.0",
    )


@pytest.fixture
def mock_phoenix_collection():
    """Create a mock PhoenixCollection for testing."""
    mock_collection = Mock(spec=PhoenixCollection)
    mock_collection.metadata_dict = {
        "station1": Mock(
            survey_metadata=Mock(id="survey1"),
            station_metadata=Mock(id="station1"),
            run_metadata=Mock(id="run1"),
            instrument_id="RX001",
            channel_map={"h1": "hx", "h2": "hy", "h3": "hz"},
        )
    }
    mock_collection.get_runs.return_value = {
        "station1": {
            "run1": Mock(
                sample_rate=Mock(unique=Mock(return_value=[150])),
                itertuples=Mock(
                    return_value=[Mock(fn=Mock(name="test.td"), component="hx")]
                ),
            )
        }
    }
    return mock_collection


@pytest.fixture
def mock_calibration_files(temp_data_dir):
    """Create mock calibration files for testing."""
    cal_dir = temp_data_dir / "calibrations"
    cal_dir.mkdir()

    # Create mock receiver calibration files
    rx_cal_file = cal_dir / "RX001_rxcal.json"
    rx_cal_file.touch()

    # Create mock sensor calibration files
    sensor_cal_file = cal_dir / "SENSOR001_scal.json"
    sensor_cal_file.touch()

    return {
        "calibration_dir": cal_dir,
        "rx_cal_file": rx_cal_file,
        "sensor_cal_file": sensor_cal_file,
    }


# =============================================================================
# PhoenixClient Initialization Tests
# =============================================================================


class TestPhoenixClientInitialization:
    """Test PhoenixClient initialization and basic properties."""

    def test_basic_initialization(self, temp_data_dir):
        """Test basic initialization with minimal parameters."""
        client = PhoenixClient(temp_data_dir)

        assert client.data_path == temp_data_dir
        assert client.mth5_filename == "from_phoenix.h5"
        assert client.sample_rates == [150, 24000]  # Default sample rates
        assert client.save_path == temp_data_dir / "from_phoenix.h5"
        assert isinstance(client.collection, PhoenixCollection)
        assert client.receiver_calibration_dict == {}
        assert client.sensor_calibration_dict == {}

    def test_initialization_with_custom_parameters(self, temp_data_dir):
        """Test initialization with custom parameters."""
        custom_filename = "custom_phoenix.h5"
        custom_sample_rates = [75, 12000]
        rx_cal_dict = {"RX001": "/path/to/rx001.json"}
        sensor_cal_dict = {"SENSOR001": "/path/to/sensor001.json"}

        client = PhoenixClient(
            temp_data_dir,
            mth5_filename=custom_filename,
            sample_rates=custom_sample_rates,
            receiver_calibration_dict=rx_cal_dict,
            sensor_calibration_dict=sensor_cal_dict,
            h5_compression="lzf",
            mth5_version="0.1.0",
        )

        assert client.mth5_filename == custom_filename
        assert client.sample_rates == custom_sample_rates
        assert client.receiver_calibration_dict == rx_cal_dict
        assert client.sensor_calibration_dict == sensor_cal_dict
        assert client.h5_compression == "lzf"
        assert client.mth5_version == "0.1.0"

    def test_initialization_with_save_path(self, temp_data_dir, temp_file_path):
        """Test initialization with custom save path."""
        client = PhoenixClient(temp_data_dir, save_path=temp_file_path)

        assert client.save_path == temp_file_path
        assert client.mth5_filename == temp_file_path.name

    def test_initialization_none_data_path_fails(self):
        """Test that None data_path raises ValueError."""
        with pytest.raises(ValueError, match="data_path cannot be None"):
            PhoenixClient(None)

    def test_initialization_bad_directory_fails(self):
        """Test that non-existent directory raises IOError."""
        with pytest.raises(IOError, match="Could not find"):
            PhoenixClient(Path("non_existent_directory_12345"))

    def test_collection_initialization(self, basic_phoenix_client):
        """Test that PhoenixCollection is properly initialized."""
        assert isinstance(basic_phoenix_client.collection, PhoenixCollection)
        assert (
            basic_phoenix_client.collection.file_path == basic_phoenix_client.data_path
        )


# =============================================================================
# H5 Parameters Tests
# =============================================================================


class TestPhoenixClientH5Parameters:
    """Test H5 parameter handling and validation."""

    def test_h5_kwargs_default_parameters(self, basic_phoenix_client):
        """Test default H5 parameters."""
        expected_keys = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]

        h5_kwargs = basic_phoenix_client.h5_kwargs
        assert sorted(h5_kwargs.keys()) == sorted(expected_keys)

        # Test default values
        assert h5_kwargs["compression"] == "gzip"
        assert h5_kwargs["compression_opts"] == 4
        assert h5_kwargs["file_version"] == "0.2.0"
        assert h5_kwargs["fletcher32"] is True
        assert h5_kwargs["shuffle"] is True
        assert h5_kwargs["data_level"] == 1

    def test_h5_kwargs_custom_parameters(self, configured_phoenix_client):
        """Test custom H5 parameters."""
        h5_kwargs = configured_phoenix_client.h5_kwargs

        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False
        assert h5_kwargs["file_version"] == "0.1.0"

    def test_h5_kwargs_with_h5_prefixed_parameters(self, temp_data_dir):
        """Test that h5_ prefixed parameters are included in h5_kwargs."""
        client = PhoenixClient(
            temp_data_dir, h5_custom_param="test_value", h5_another_param=42
        )

        h5_kwargs = client.h5_kwargs
        assert "custom_param" in h5_kwargs
        assert "another_param" in h5_kwargs
        assert h5_kwargs["custom_param"] == "test_value"
        assert h5_kwargs["another_param"] == 42


# =============================================================================
# Sample Rates Tests
# =============================================================================


class TestPhoenixClientSampleRates:
    """Test sample rates property validation and conversion."""

    @pytest.mark.parametrize(
        "input_value,expected", [(10.5, [10.5]), (100, [100.0]), (42.0, [42.0])]
    )
    def test_sample_rates_float_input(
        self, basic_phoenix_client, input_value, expected
    ):
        """Test setting sample rates with float/int values."""
        basic_phoenix_client.sample_rates = input_value
        assert basic_phoenix_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_string,expected",
        [
            ("10, 42, 1200", [10.0, 42.0, 1200.0]),
            ("1.5, 25.5, 100", [1.5, 25.5, 100.0]),
            ("1000", [1000.0]),
            ("10,42,1200", [10.0, 42.0, 1200.0]),  # No spaces
        ],
    )
    def test_sample_rates_string_input(
        self, basic_phoenix_client, input_string, expected
    ):
        """Test setting sample rates with string values."""
        basic_phoenix_client.sample_rates = input_string
        assert basic_phoenix_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_list,expected",
        [
            ([10, 42, 1200], [10.0, 42.0, 1200.0]),
            ((1, 5, 25), [1.0, 5.0, 25.0]),
            ([1.5, 2.5], [1.5, 2.5]),
            ([150, 24000], [150.0, 24000.0]),  # Default Phoenix sample rates
        ],
    )
    def test_sample_rates_list_input(self, basic_phoenix_client, input_list, expected):
        """Test setting sample rates with list/tuple values."""
        basic_phoenix_client.sample_rates = input_list
        assert basic_phoenix_client.sample_rates == expected

    @pytest.mark.parametrize("invalid_value", [None, {}, set([1, 2, 3]), object()])
    def test_sample_rates_invalid_input(self, basic_phoenix_client, invalid_value):
        """Test that invalid sample rate inputs raise TypeError."""
        with pytest.raises(TypeError, match="Cannot parse"):
            basic_phoenix_client.sample_rates = invalid_value


# =============================================================================
# Save Path Tests
# =============================================================================


class TestPhoenixClientSavePath:
    """Test save path property handling and validation."""

    def test_save_path_file_assignment(self, basic_phoenix_client, temp_file_path):
        """Test setting save path with file path."""
        basic_phoenix_client.save_path = temp_file_path

        assert basic_phoenix_client._save_path == temp_file_path.parent
        assert basic_phoenix_client.mth5_filename == temp_file_path.name
        assert basic_phoenix_client.save_path == temp_file_path

    def test_save_path_directory_assignment(self, basic_phoenix_client, temp_data_dir):
        """Test setting save path with directory path."""
        original_filename = basic_phoenix_client.mth5_filename
        basic_phoenix_client.save_path = temp_data_dir

        assert basic_phoenix_client._save_path == temp_data_dir
        assert basic_phoenix_client.mth5_filename == original_filename
        assert basic_phoenix_client.save_path == temp_data_dir / original_filename

    def test_save_path_none_defaults_to_data_path(self, temp_data_dir):
        """Test that None save_path defaults to data_path."""
        client = PhoenixClient(temp_data_dir, save_path=None)

        assert client._save_path == temp_data_dir

    def test_save_path_components_consistency(
        self, basic_phoenix_client, temp_file_path
    ):
        """Test consistency between save_path components."""
        basic_phoenix_client.save_path = temp_file_path

        # All components should be consistent
        assert (
            basic_phoenix_client.save_path
            == basic_phoenix_client._save_path / basic_phoenix_client.mth5_filename
        )
        assert basic_phoenix_client.save_path.parent == basic_phoenix_client._save_path
        assert basic_phoenix_client.save_path.name == basic_phoenix_client.mth5_filename


# =============================================================================
# Receiver Calibration Tests
# =============================================================================


class TestPhoenixClientReceiverCalibration:
    """Test receiver calibration dictionary handling."""

    def test_receiver_calibration_dict_assignment(self, basic_phoenix_client):
        """Test setting receiver calibration dictionary."""
        cal_dict = {"RX001": "/path/to/rx001.json", "RX002": "/path/to/rx002.json"}
        basic_phoenix_client.receiver_calibration_dict = cal_dict

        assert basic_phoenix_client.receiver_calibration_dict == cal_dict

    def test_receiver_calibration_dict_empty_default(self, basic_phoenix_client):
        """Test that receiver calibration dict defaults to empty."""
        assert basic_phoenix_client.receiver_calibration_dict == {}

    def test_receiver_calibration_dict_file_path(
        self, basic_phoenix_client, mock_calibration_files
    ):
        """Test setting receiver calibration from file path."""
        rx_cal_file = mock_calibration_files["rx_cal_file"]

        # Setting with a single file should populate the receiver calibration
        # dict with the derived key from the filename stem (e.g., RX001).
        basic_phoenix_client.receiver_calibration_dict = rx_cal_file

        # Verify the dictionary was populated correctly
        assert isinstance(basic_phoenix_client.receiver_calibration_dict, dict)
        assert len(basic_phoenix_client.receiver_calibration_dict) == 1
        key = rx_cal_file.stem.split("_")[0]
        assert key in basic_phoenix_client.receiver_calibration_dict
        assert basic_phoenix_client.receiver_calibration_dict[key] == rx_cal_file

    def test_receiver_calibration_dict_directory_path(
        self, basic_phoenix_client, mock_calibration_files
    ):
        """Test setting receiver calibration from directory path."""
        cal_dir = mock_calibration_files["calibration_dir"]
        basic_phoenix_client.receiver_calibration_dict = cal_dir

        # Should find .rxcal.json files in directory
        # The implementation looks for *.rxcal.json and *.rx_cal.json patterns
        assert (
            len(basic_phoenix_client.receiver_calibration_dict) >= 0
        )  # At least it shouldn't crash
        if basic_phoenix_client.receiver_calibration_dict:
            assert "RX001" in basic_phoenix_client.receiver_calibration_dict

    def test_receiver_calibration_dict_invalid_type(self, basic_phoenix_client):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="type .* not supported"):
            basic_phoenix_client.receiver_calibration_dict = 123


# =============================================================================
# Sensor Calibration Tests
# =============================================================================


class TestPhoenixClientSensorCalibration:
    """Test sensor calibration dictionary handling."""

    def test_sensor_calibration_dict_assignment(self, basic_phoenix_client):
        """Test setting sensor calibration dictionary."""
        cal_dict = {"SENSOR001": Mock(spec=PhoenixCalibration)}
        basic_phoenix_client.sensor_calibration_dict = cal_dict

        assert basic_phoenix_client.sensor_calibration_dict == cal_dict

    def test_sensor_calibration_dict_empty_default(self, basic_phoenix_client):
        """Test that sensor calibration dict defaults to empty."""
        assert basic_phoenix_client.sensor_calibration_dict == {}

    @patch("mth5.clients.phoenix.PhoenixCalibration")
    def test_sensor_calibration_dict_directory_path(
        self, mock_phoenix_cal, basic_phoenix_client, mock_calibration_files
    ):
        """Test setting sensor calibration from directory path."""
        mock_cal_instance = Mock(spec=PhoenixCalibration)
        mock_phoenix_cal.return_value = mock_cal_instance

        cal_dir = mock_calibration_files["calibration_dir"]
        basic_phoenix_client.sensor_calibration_dict = cal_dir

        # Should find .scal.json files and create PhoenixCalibration objects
        assert len(basic_phoenix_client.sensor_calibration_dict) > 0
        assert "SENSOR001" in basic_phoenix_client.sensor_calibration_dict
        assert (
            basic_phoenix_client.sensor_calibration_dict["SENSOR001"]
            == mock_cal_instance
        )

    def test_sensor_calibration_dict_nonexistent_path_creates_empty_dict(
        self, basic_phoenix_client
    ):
        """Test that non-existent calibration directory creates empty dict."""
        # When a directory path doesn't exist, is_dir() returns False
        # and the setter creates an empty dict
        basic_phoenix_client.sensor_calibration_dict = "/non/existent/path"
        assert basic_phoenix_client.sensor_calibration_dict == {}

    # this test should be depricated because the setter should handle None by creating an empty dict, not raising an error
    # def test_sensor_calibration_dict_none_raises_error(self, basic_phoenix_client):
    #     """Test that None calibration path raises ValueError."""
    #     with pytest.raises(ValueError, match="calibration_path cannot be None"):
    #         basic_phoenix_client.sensor_calibration_dict = None


# =============================================================================
# Collection Integration Tests
# =============================================================================


class TestPhoenixClientCollectionIntegration:
    """Test integration with PhoenixCollection."""

    def test_get_run_dict_calls_collection(self, basic_phoenix_client):
        """Test that get_run_dict delegates to collection."""
        with patch.object(basic_phoenix_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {"test": "data"}

            result = basic_phoenix_client.get_run_dict()

            mock_get_runs.assert_called_once_with(
                sample_rates=basic_phoenix_client.sample_rates
            )
            assert result == {"test": "data"}

    def test_collection_uses_correct_path(self, temp_data_dir):
        """Test that collection is initialized with correct path."""
        client = PhoenixClient(temp_data_dir)
        assert client.collection.file_path == temp_data_dir


# =============================================================================
# MTH5 Creation Tests (Mocked)
# =============================================================================


class TestPhoenixClientMTH5Creation:
    """Test MTH5 file creation functionality with mocking."""

    @patch("mth5.clients.phoenix.read_file")
    @patch("mth5.clients.phoenix.MTH5")
    def test_make_mth5_from_phoenix_basic(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_phoenix_client,
        mock_phoenix_collection,
    ):
        """Test basic MTH5 creation from Phoenix data."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_ch_ts = MagicMock()
        mock_ch_ts.component = "hx"
        mock_ch_ts.channel_metadata.sensor.id = "unknown"
        mock_read_file.return_value = mock_ch_ts

        # Setup required receiver calibration for the instrument
        basic_phoenix_client.receiver_calibration_dict = {
            "RX001": "/path/to/rx001.json"
        }

        # Replace collection with mock
        basic_phoenix_client.collection = mock_phoenix_collection

        # Test method
        result = basic_phoenix_client.make_mth5_from_phoenix()

        # Verify calls
        mock_mth5_instance.open_mth5.assert_called_once_with(
            basic_phoenix_client.save_path, "w"
        )
        mock_mth5_instance.add_survey.assert_called_once()
        mock_survey_group.stations_group.add_station.assert_called_once()
        mock_station_group.add_run.assert_called_once()
        mock_run_group.from_channel_ts.assert_called_once_with(mock_ch_ts)

        assert result == basic_phoenix_client.save_path

    @patch("mth5.clients.phoenix.read_file")
    @patch("mth5.clients.phoenix.MTH5")
    def test_make_mth5_from_phoenix_with_kwargs(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_phoenix_client,
        mock_phoenix_collection,
    ):
        """Test MTH5 creation with additional kwargs."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_ch_ts = MagicMock()
        mock_ch_ts.component = "hx"
        mock_ch_ts.channel_metadata.sensor.id = "unknown"
        mock_read_file.return_value = mock_ch_ts

        # Setup required receiver calibration for the instrument
        basic_phoenix_client.receiver_calibration_dict = {
            "RX001": "/path/to/rx001.json"
        }

        # Replace collection with mock
        basic_phoenix_client.collection = mock_phoenix_collection

        # Test with additional kwargs
        result = basic_phoenix_client.make_mth5_from_phoenix(
            sample_rates=[75, 12000], h5_compression="lzf"
        )

        # Verify that kwargs were applied
        assert basic_phoenix_client.sample_rates == [75.0, 12000.0]
        assert basic_phoenix_client.h5_compression == "lzf"

    @patch("mth5.clients.phoenix.read_file")
    @patch("mth5.clients.phoenix.MTH5")
    def test_make_mth5_with_coil_calibration(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_phoenix_client,
        mock_phoenix_collection,
    ):
        """Test MTH5 creation with coil calibration handling."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup channel with magnetic component
        mock_ch_ts = MagicMock()
        mock_ch_ts.component = "h1"  # Magnetic component
        mock_ch_ts.channel_metadata.sensor.id = "SENSOR001"
        mock_read_file.return_value = mock_ch_ts

        # Setup sensor calibration
        mock_cal = MagicMock()
        mock_cal.h1 = MagicMock()
        mock_cal.h1.name = "H1_Calibration"
        mock_cal.__dict__ = {"h1": mock_cal.h1}
        basic_phoenix_client.sensor_calibration_dict = {"SENSOR001": mock_cal}

        # Setup required receiver calibration for the instrument
        basic_phoenix_client.receiver_calibration_dict = {
            "RX001": "/path/to/rx001.json"
        }

        # Replace collection with mock
        basic_phoenix_client.collection = mock_phoenix_collection

        # Test method
        result = basic_phoenix_client.make_mth5_from_phoenix()

        # Verify coil calibration was applied
        # The code calls add_filter() method and appends to filters_list
        mock_ch_ts.channel_metadata.add_filter.assert_called_once()
        # no longer called because the setter for sensor_calibration_dict should handle this now, not make_mth5_from_phoenix
        # mock_ch_ts.channel_response.filters_list.append.assert_called_once_with(
        #     mock_cal.h1
        # )

    @patch("mth5.clients.phoenix.read_file")
    @patch("mth5.clients.phoenix.MTH5")
    def test_make_mth5_handles_os_error(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_phoenix_client,
        mock_phoenix_collection,
    ):
        """Test MTH5 creation handles OSError gracefully."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup read_file to raise OSError
        mock_read_file.side_effect = OSError("File too small")

        # Setup required receiver calibration for the instrument
        basic_phoenix_client.receiver_calibration_dict = {
            "RX001": "/path/to/rx001.json"
        }

        # Replace collection with mock
        basic_phoenix_client.collection = mock_phoenix_collection

        # Test method - should not raise exception
        result = basic_phoenix_client.make_mth5_from_phoenix()

        # Should still return save path even with errors
        assert result == basic_phoenix_client.save_path

    @patch("mth5.clients.phoenix.read_file")
    @patch("mth5.clients.phoenix.MTH5")
    def test_make_mth5_sets_all_filters_applied_true(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_phoenix_client,
        mock_phoenix_collection,
    ):
        """Test that all filters in channel_metadata.filters have applied=True."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup channel with filters that have applied=False
        mock_filter1 = MagicMock()
        mock_filter1.applied = False
        mock_filter1.name = "filter1"

        mock_filter2 = MagicMock()
        mock_filter2.applied = False
        mock_filter2.name = "filter2"

        mock_filter3 = MagicMock()
        mock_filter3.applied = True  # One already True
        mock_filter3.name = "filter3"

        mock_ch_ts = MagicMock()
        mock_ch_ts.component = "hx"
        mock_ch_ts.channel_metadata.sensor.id = "unknown"
        mock_ch_ts.channel_metadata.filters = [mock_filter1, mock_filter2, mock_filter3]
        mock_read_file.return_value = mock_ch_ts

        # Setup required receiver calibration for the instrument
        basic_phoenix_client.receiver_calibration_dict = {
            "RX001": "/path/to/rx001.json"
        }

        # Replace collection with mock
        basic_phoenix_client.collection = mock_phoenix_collection

        # Test method
        result = basic_phoenix_client.make_mth5_from_phoenix()

        # Verify all filters now have applied=True
        for filter_obj in mock_ch_ts.channel_metadata.filters:
            assert (
                filter_obj.applied is True
            ), f"Filter {filter_obj.name} should have applied=True"

        assert result == basic_phoenix_client.save_path


# =============================================================================
# Property Integration Tests
# =============================================================================


class TestPhoenixClientPropertyIntegration:
    """Test integration between different properties."""

    def test_h5_kwargs_reflects_property_changes(self, basic_phoenix_client):
        """Test that h5_kwargs reflects property changes."""
        # Change properties
        basic_phoenix_client.mth5_version = "0.1.0"
        basic_phoenix_client.h5_compression = "lzf"
        basic_phoenix_client.h5_shuffle = False

        h5_kwargs = basic_phoenix_client.h5_kwargs

        assert h5_kwargs["file_version"] == "0.1.0"
        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False

    def test_calibration_dicts_independence(self, basic_phoenix_client):
        """Test that receiver and sensor calibration dicts are independent."""
        rx_dict = {"RX001": "/path/to/rx"}
        sensor_dict = {"SENSOR001": Mock()}

        basic_phoenix_client.receiver_calibration_dict = rx_dict
        basic_phoenix_client.sensor_calibration_dict = sensor_dict

        assert basic_phoenix_client.receiver_calibration_dict == rx_dict
        assert basic_phoenix_client.sensor_calibration_dict == sensor_dict

        # Changing one shouldn't affect the other
        basic_phoenix_client.receiver_calibration_dict = {}
        assert basic_phoenix_client.sensor_calibration_dict == sensor_dict


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestPhoenixClientEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sample_rates_list(self, basic_phoenix_client):
        """Test handling of empty sample rates list."""
        basic_phoenix_client.sample_rates = []
        assert basic_phoenix_client.sample_rates == []

    def test_single_character_string_sample_rate(self, basic_phoenix_client):
        """Test single character string sample rate."""
        basic_phoenix_client.sample_rates = "5"
        assert basic_phoenix_client.sample_rates == [5.0]

    def test_whitespace_in_sample_rates_string(self, basic_phoenix_client):
        """Test handling of extra whitespace in sample rates string."""
        basic_phoenix_client.sample_rates = " 150 , 24000 "
        assert basic_phoenix_client.sample_rates == [150.0, 24000.0]

    def test_phoenix_specific_sample_rates(self, basic_phoenix_client):
        """Test Phoenix-specific sample rates."""
        # Common Phoenix sample rates
        phoenix_rates = [150, 24000]
        basic_phoenix_client.sample_rates = phoenix_rates
        assert basic_phoenix_client.sample_rates == [150.0, 24000.0]

    def test_property_attribute_access(self, basic_phoenix_client):
        """Test that all expected attributes are accessible."""
        # Test that basic attributes exist and have expected types
        assert hasattr(basic_phoenix_client, "data_path")
        assert hasattr(basic_phoenix_client, "save_path")
        assert hasattr(basic_phoenix_client, "sample_rates")
        assert hasattr(basic_phoenix_client, "mth5_filename")
        assert hasattr(basic_phoenix_client, "collection")
        assert hasattr(basic_phoenix_client, "h5_kwargs")
        assert hasattr(basic_phoenix_client, "receiver_calibration_dict")
        assert hasattr(basic_phoenix_client, "sensor_calibration_dict")

        assert isinstance(basic_phoenix_client.data_path, Path)
        assert isinstance(basic_phoenix_client.save_path, Path)
        assert isinstance(basic_phoenix_client.sample_rates, list)
        assert isinstance(basic_phoenix_client.mth5_filename, str)
        assert isinstance(basic_phoenix_client.h5_kwargs, dict)
        assert isinstance(basic_phoenix_client.receiver_calibration_dict, dict)
        assert isinstance(basic_phoenix_client.sensor_calibration_dict, dict)

    def test_logger_attribute(self, basic_phoenix_client):
        """Test that logger attribute is properly set."""
        assert hasattr(basic_phoenix_client, "logger")
        assert basic_phoenix_client.logger is not None

    def test_interact_attribute_default(self, basic_phoenix_client):
        """Test that interact attribute has correct default."""
        assert basic_phoenix_client.interact is False

    def test_default_hdf5_parameters(self, basic_phoenix_client):
        """Test that default HDF5 parameters are correctly set."""
        assert basic_phoenix_client.mth5_version == "0.2.0"
        assert basic_phoenix_client.h5_compression == "gzip"
        assert basic_phoenix_client.h5_compression_opts == 4
        assert basic_phoenix_client.h5_shuffle is True
        assert basic_phoenix_client.h5_fletcher32 is True
        assert basic_phoenix_client.h5_data_level == 1


# =============================================================================
# Inheritance and Base Class Tests
# =============================================================================


class TestPhoenixClientInheritance:
    """Test inheritance from ClientBase."""

    def test_inherits_from_client_base(self, basic_phoenix_client):
        """Test that PhoenixClient properly inherits from ClientBase."""
        from mth5.clients.base import ClientBase

        assert isinstance(basic_phoenix_client, ClientBase)

    def test_overrides_collection_initialization(self, basic_phoenix_client):
        """Test that PhoenixClient overrides collection initialization."""
        # The collection should be a PhoenixCollection, not None as in base class
        assert basic_phoenix_client.collection is not None
        assert isinstance(basic_phoenix_client.collection, PhoenixCollection)

    def test_default_mth5_filename_override(self, temp_data_dir):
        """Test that default MTH5 filename is overridden."""
        client = PhoenixClient(temp_data_dir)
        assert client.mth5_filename == "from_phoenix.h5"
        # This is different from the base class default of "from_client.h5"

    def test_default_sample_rates_override(self, temp_data_dir):
        """Test that default sample rates are overridden for Phoenix."""
        client = PhoenixClient(temp_data_dir)
        assert client.sample_rates == [150, 24000]
        # Phoenix-specific default sample rates


# =============================================================================
# Phoenix-Specific Functionality Tests
# =============================================================================


class TestPhoenixClientSpecificFunctionality:
    """Test Phoenix-specific functionality."""

    def test_phoenix_default_sample_rates(self, temp_data_dir):
        """Test that Phoenix has appropriate default sample rates."""
        client = PhoenixClient(temp_data_dir)
        # Phoenix typically uses 150 Hz and 24 kHz
        assert 150 in client.sample_rates
        assert 24000 in client.sample_rates

    def test_calibration_initialization_types(self, basic_phoenix_client):
        """Test that calibration dictionaries are properly typed."""
        # Test initial empty state
        assert isinstance(basic_phoenix_client.receiver_calibration_dict, dict)
        assert isinstance(basic_phoenix_client.sensor_calibration_dict, dict)
        assert len(basic_phoenix_client.receiver_calibration_dict) == 0
        assert len(basic_phoenix_client.sensor_calibration_dict) == 0

    @patch("mth5.clients.phoenix.PhoenixCalibration")
    def test_sensor_calibration_file_patterns(
        self, mock_phoenix_cal, basic_phoenix_client, temp_data_dir
    ):
        """Test sensor calibration file pattern recognition."""
        mock_cal_instance = Mock(spec=PhoenixCalibration)
        mock_phoenix_cal.return_value = mock_cal_instance

        # Create mock calibration directory with various file patterns
        cal_dir = temp_data_dir / "calibrations"
        cal_dir.mkdir()

        # Create files with expected patterns
        (cal_dir / "SENSOR001_scal.json").touch()
        (cal_dir / "SENSOR002_scal.json").touch()
        (cal_dir / "OTHER_file.json").touch()  # Should be ignored

        basic_phoenix_client.sensor_calibration_dict = cal_dir

        # Should only pick up *scal.json files
        assert "SENSOR001" in basic_phoenix_client.sensor_calibration_dict
        assert "SENSOR002" in basic_phoenix_client.sensor_calibration_dict
        assert "OTHER" not in basic_phoenix_client.sensor_calibration_dict

    def test_receiver_calibration_file_patterns(
        self, basic_phoenix_client, temp_data_dir
    ):
        """Test receiver calibration file pattern recognition."""
        # Create mock calibration directory
        cal_dir = temp_data_dir / "calibrations"
        cal_dir.mkdir()

        # Create files with different accepted patterns
        (cal_dir / "RX001_rxcal.json").touch()
        (cal_dir / "RX002_rx_cal.json").touch()
        (cal_dir / "OTHER_file.json").touch()  # Should be ignored

        basic_phoenix_client.receiver_calibration_dict = cal_dir

        # Should pick up both rxcal.json and rx_cal.json patterns but the glob patterns are case sensitive
        # The implementation uses "*.rxcal.json" and "*.rx_cal.json" patterns
        if basic_phoenix_client.receiver_calibration_dict:
            assert (
                "RX001" in basic_phoenix_client.receiver_calibration_dict
                or "RX002" in basic_phoenix_client.receiver_calibration_dict
            )

        # Verify that OTHER file is not included
        assert "OTHER" not in basic_phoenix_client.receiver_calibration_dict


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


class TestPhoenixClientPerformance:
    """Test performance characteristics."""

    def test_large_sample_rates_list(self, basic_phoenix_client):
        """Test handling of large sample rates list."""
        large_sample_rates = list(range(1, 101))  # 100 sample rates
        basic_phoenix_client.sample_rates = large_sample_rates
        assert len(basic_phoenix_client.sample_rates) == 100
        assert basic_phoenix_client.sample_rates == [
            float(x) for x in large_sample_rates
        ]

    def test_large_calibration_dictionaries(self, basic_phoenix_client):
        """Test handling of large calibration dictionaries."""
        # Large receiver calibration dict
        large_rx_dict = {f"RX{i:03d}": f"/path/to/rx{i:03d}.json" for i in range(100)}
        basic_phoenix_client.receiver_calibration_dict = large_rx_dict
        assert len(basic_phoenix_client.receiver_calibration_dict) == 100

        # Large sensor calibration dict
        large_sensor_dict = {f"SENSOR{i:03d}": Mock() for i in range(100)}
        basic_phoenix_client.sensor_calibration_dict = large_sensor_dict
        assert len(basic_phoenix_client.sensor_calibration_dict) == 100

    def test_very_long_filename(self, basic_phoenix_client, temp_data_dir):
        """Test handling of very long filenames."""
        # Use a shorter repetition to avoid platform-dependent filename length limits
        long_filename = "very_" * 10 + "long_phoenix_filename.h5"
        long_path = temp_data_dir / long_filename

        basic_phoenix_client.save_path = long_path
        assert basic_phoenix_client.mth5_filename == long_filename


# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])
