# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for LEMI424Client with fixtures, parametrization,
and additional coverage for untested functionality.

Created by modernizing test_lemi424.py with pytest patterns.

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
from mt_io.lemi import LEMICollection

from mth5.clients.lemi424 import LEMI424Client


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
def basic_lemi_client(temp_data_dir):
    """Create a basic LEMI424Client instance for testing."""
    return LEMI424Client(temp_data_dir, **{"h5_mode": "w", "h5_driver": "sec2"})


@pytest.fixture
def configured_lemi_client(temp_data_dir, temp_file_path):
    """Create a configured LEMI424Client instance with custom settings."""
    client = LEMI424Client(
        temp_data_dir,
        save_path=temp_file_path.parent,
        mth5_filename=temp_file_path.name,
        h5_compression="lzf",
        h5_shuffle=False,
        mth5_version="0.1.0",
    )
    # Set sample_rates after initialization since it's hardcoded in constructor
    client.sample_rates = [1, 10, 100]
    return client


@pytest.fixture
def mock_lemi_collection():
    """Create a mock LEMICollection for testing."""
    mock_collection = Mock(spec=LEMICollection)
    mock_collection.survey_id = "test_survey"
    mock_collection.station_id = "test_station"
    mock_collection.calibration_dict = {}
    mock_collection.get_runs.return_value = {
        "test_station": {
            "sr1_001": Mock(
                fn=Mock(to_list=Mock(return_value=["file1.txt", "file2.txt"]))
            )
        }
    }
    return mock_collection


@pytest.fixture
def mock_lemi_collection_with_calibration(temp_data_dir):
    """Create a mock LEMICollection with populated calibration_dict."""
    mock_collection = Mock(spec=LEMICollection)
    mock_collection.survey_id = "test_survey"
    mock_collection.station_id = "test_station"

    # Create calibration dict with magnetic channels only
    mock_collection.calibration_dict = {
        "bx": str(temp_data_dir / "lemi-bx.sr.json"),
        "by": str(temp_data_dir / "lemi-by.sr.json"),
        "bz": str(temp_data_dir / "lemi-bz.sr.json"),
    }

    mock_collection.get_runs.return_value = {
        "test_station": {
            "sr1_001": Mock(
                fn=Mock(to_list=Mock(return_value=["file1.txt", "file2.txt"]))
            )
        }
    }
    return mock_collection


# =============================================================================
# LEMI424Client Initialization Tests
# =============================================================================


class TestLEMI424ClientInitialization:
    """Test LEMI424Client initialization and basic properties."""

    def test_basic_initialization(self, temp_data_dir):
        """Test basic initialization with minimal parameters."""
        client = LEMI424Client(temp_data_dir)

        assert client.data_path == temp_data_dir
        assert client.mth5_filename == "from_lemi424.h5"
        assert client.sample_rates == [1]
        assert client.save_path == temp_data_dir / "from_lemi424.h5"
        assert isinstance(client.collection, LEMICollection)

    def test_initialization_with_custom_parameters(self, temp_data_dir):
        """Test initialization with custom parameters."""
        custom_filename = "custom_lemi.h5"

        client = LEMI424Client(
            temp_data_dir,
            mth5_filename=custom_filename,
            h5_compression="lzf",
            mth5_version="0.1.0",
        )

        assert client.mth5_filename == custom_filename
        # Note: sample_rates is fixed at [1] in LEMI424Client constructor
        assert client.sample_rates == [1]
        assert client.h5_compression == "lzf"
        assert client.mth5_version == "0.1.0"

        # Test that we can change sample_rates after initialization
        client.sample_rates = [1, 10, 100]
        assert client.sample_rates == [1.0, 10.0, 100.0]

    def test_initialization_with_save_path(self, temp_data_dir, temp_file_path):
        """Test initialization with custom save path."""
        client = LEMI424Client(temp_data_dir, save_path=temp_file_path)

        assert client.save_path == temp_file_path
        assert client.mth5_filename == temp_file_path.name

    def test_initialization_none_data_path_fails(self):
        """Test that None data_path raises ValueError."""
        with pytest.raises(ValueError, match="data_path cannot be None"):
            LEMI424Client(None)

    def test_initialization_bad_directory_fails(self):
        """Test that non-existent directory raises IOError."""
        with pytest.raises(IOError, match="Could not find"):
            LEMI424Client(Path("non_existent_directory_12345"))

    def test_collection_initialization(self, basic_lemi_client):
        """Test that LEMICollection is properly initialized."""
        assert isinstance(basic_lemi_client.collection, LEMICollection)
        assert basic_lemi_client.collection.file_path == basic_lemi_client.data_path


# =============================================================================
# H5 Parameters Tests
# =============================================================================


class TestLEMI424ClientH5Parameters:
    """Test H5 parameter handling and validation."""

    def test_h5_kwargs_default_parameters(self, basic_lemi_client):
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

        h5_kwargs = basic_lemi_client.h5_kwargs
        assert sorted(h5_kwargs.keys()) == sorted(expected_keys)

        # Test default values
        assert h5_kwargs["compression"] == "gzip"
        assert h5_kwargs["compression_opts"] == 4
        assert h5_kwargs["file_version"] == "0.2.0"
        assert h5_kwargs["fletcher32"] is True
        assert h5_kwargs["shuffle"] is True
        assert h5_kwargs["data_level"] == 1

    def test_h5_kwargs_custom_parameters(self, configured_lemi_client):
        """Test custom H5 parameters."""
        h5_kwargs = configured_lemi_client.h5_kwargs

        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False
        assert h5_kwargs["file_version"] == "0.1.0"

    def test_h5_kwargs_with_h5_prefixed_parameters(self, temp_data_dir):
        """Test that h5_ prefixed parameters are included in h5_kwargs."""
        client = LEMI424Client(
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


class TestLEMI424ClientSampleRates:
    """Test sample rates property validation and conversion."""

    @pytest.mark.parametrize(
        "input_value,expected", [(10.5, [10.5]), (100, [100.0]), (42.0, [42.0])]
    )
    def test_sample_rates_float_input(self, basic_lemi_client, input_value, expected):
        """Test setting sample rates with float/int values."""
        basic_lemi_client.sample_rates = input_value
        assert basic_lemi_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_string,expected",
        [
            ("10, 42, 1200", [10.0, 42.0, 1200.0]),
            ("1.5, 25.5, 100", [1.5, 25.5, 100.0]),
            ("1000", [1000.0]),
            ("10,42,1200", [10.0, 42.0, 1200.0]),  # No spaces
        ],
    )
    def test_sample_rates_string_input(self, basic_lemi_client, input_string, expected):
        """Test setting sample rates with string values."""
        basic_lemi_client.sample_rates = input_string
        assert basic_lemi_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_list,expected",
        [
            ([10, 42, 1200], [10.0, 42.0, 1200.0]),
            ((1, 5, 25), [1.0, 5.0, 25.0]),
            ([1.5, 2.5], [1.5, 2.5]),
        ],
    )
    def test_sample_rates_list_input(self, basic_lemi_client, input_list, expected):
        """Test setting sample rates with list/tuple values."""
        basic_lemi_client.sample_rates = input_list
        assert basic_lemi_client.sample_rates == expected

    @pytest.mark.parametrize("invalid_value", [None, {}, set([1, 2, 3]), object()])
    def test_sample_rates_invalid_input(self, basic_lemi_client, invalid_value):
        """Test that invalid sample rate inputs raise TypeError."""
        with pytest.raises(TypeError, match="Cannot parse"):
            basic_lemi_client.sample_rates = invalid_value


# =============================================================================
# Save Path Tests
# =============================================================================


class TestLEMI424ClientSavePath:
    """Test save path property handling and validation."""

    def test_save_path_file_assignment(self, basic_lemi_client, temp_file_path):
        """Test setting save path with file path."""
        basic_lemi_client.save_path = temp_file_path

        assert basic_lemi_client._save_path == temp_file_path.parent
        assert basic_lemi_client.mth5_filename == temp_file_path.name
        assert basic_lemi_client.save_path == temp_file_path

    def test_save_path_directory_assignment(self, basic_lemi_client, temp_data_dir):
        """Test setting save path with directory path."""
        original_filename = basic_lemi_client.mth5_filename
        basic_lemi_client.save_path = temp_data_dir

        assert basic_lemi_client._save_path == temp_data_dir
        assert basic_lemi_client.mth5_filename == original_filename
        assert basic_lemi_client.save_path == temp_data_dir / original_filename

    def test_save_path_none_defaults_to_data_path(self, temp_data_dir):
        """Test that None save_path defaults to data_path."""
        client = LEMI424Client(temp_data_dir, save_path=None)

        assert client._save_path == temp_data_dir

    def test_save_path_nonexistent_file_creates_directory(
        self, basic_lemi_client, temp_data_dir
    ):
        """Test that setting non-existent file path creates parent directory."""
        new_dir = temp_data_dir / "new_subdir"
        new_file = new_dir / "new_file.h5"

        basic_lemi_client.save_path = new_file

        assert new_dir.exists()
        assert basic_lemi_client._save_path == new_dir
        assert basic_lemi_client.mth5_filename == "new_file.h5"

    def test_save_path_nonexistent_directory_creates_it(
        self, basic_lemi_client, temp_data_dir
    ):
        """Test that setting non-existent directory creates it."""
        new_dir = temp_data_dir / "new_subdir"

        basic_lemi_client.save_path = new_dir

        assert new_dir.exists()
        assert basic_lemi_client._save_path == new_dir


# =============================================================================
# Data Path Tests
# =============================================================================


class TestLEMI424ClientDataPath:
    """Test data path property handling and validation."""

    def test_data_path_valid_directory(self, temp_data_dir):
        """Test setting valid data path."""
        client = LEMI424Client(temp_data_dir)
        assert client.data_path == temp_data_dir

    def test_data_path_string_conversion(self, temp_data_dir):
        """Test that string data path is converted to Path."""
        client = LEMI424Client(str(temp_data_dir))
        assert client.data_path == temp_data_dir
        assert isinstance(client.data_path, Path)

    def test_data_path_nonexistent_raises_error(self):
        """Test that non-existent data path raises IOError."""
        with pytest.raises(IOError, match="Could not find"):
            LEMI424Client("/non/existent/path")

    def test_data_path_none_raises_error(self):
        """Test that None data path raises ValueError."""
        with pytest.raises(ValueError, match="data_path cannot be None"):
            LEMI424Client(None)


# =============================================================================
# Collection Integration Tests
# =============================================================================


class TestLEMI424ClientCollectionIntegration:
    """Test integration with LEMICollection."""

    def test_get_run_dict_calls_collection(self, basic_lemi_client):
        """Test that get_run_dict delegates to collection."""
        with patch.object(basic_lemi_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {"test": "data"}

            result = basic_lemi_client.get_run_dict()

            mock_get_runs.assert_called_once_with(
                sample_rates=basic_lemi_client.sample_rates
            )
            assert result == {"test": "data"}

    def test_collection_uses_correct_path(self, temp_data_dir):
        """Test that collection is initialized with correct path."""
        client = LEMI424Client(temp_data_dir)
        assert client.collection.file_path == temp_data_dir


# =============================================================================
# MTH5 Creation Tests (Mocked)
# =============================================================================


class TestLEMI424ClientMTH5Creation:
    """Test MTH5 file creation functionality with mocking."""

    @patch("mth5.clients.lemi424.read_file")
    @patch("mth5.clients.lemi424.MTH5")
    def test_make_mth5_from_lemi424_basic(
        self, mock_mth5_class, mock_read_file, basic_lemi_client, mock_lemi_collection
    ):
        """Test basic MTH5 creation from LEMI424 data."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_run_ts = MagicMock()
        mock_run_ts.run_metadata.id = "sr1_001"
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        # Replace collection with mock
        basic_lemi_client.collection = mock_lemi_collection

        # Test method
        result = basic_lemi_client.make_mth5_from_lemi424("test_survey", "test_station")

        # Verify calls
        mock_mth5_instance.open_mth5.assert_called_once_with(
            basic_lemi_client.save_path, "w"
        )
        mock_mth5_instance.add_survey.assert_called_once_with("test_survey")
        mock_survey_group.stations_group.add_station.assert_called_once_with(
            "test_station"
        )
        mock_station_group.add_run.assert_called_once_with("sr1_001")
        mock_run_group.from_runts.assert_called_once_with(mock_run_ts)

        assert result == basic_lemi_client.save_path

    @patch("mth5.clients.lemi424.read_file")
    @patch("mth5.clients.lemi424.MTH5")
    def test_make_mth5_from_lemi424_with_kwargs(
        self, mock_mth5_class, mock_read_file, basic_lemi_client, mock_lemi_collection
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

        mock_run_ts = MagicMock()
        mock_run_ts.run_metadata.id = "sr1_001"
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        # Replace collection with mock
        basic_lemi_client.collection = mock_lemi_collection

        # Test with additional kwargs
        result = basic_lemi_client.make_mth5_from_lemi424(
            "test_survey", "test_station", h5_compression="lzf"
        )

        # Test setting sample_rates separately since it can't be passed in constructor
        basic_lemi_client.sample_rates = [1, 10]

        # Verify that kwargs were applied
        assert basic_lemi_client.h5_compression == "lzf"
        assert basic_lemi_client.sample_rates == [1.0, 10.0]

    @patch("mth5.clients.lemi424.read_file")
    @patch("mth5.clients.lemi424.MTH5")
    def test_make_mth5_collection_survey_station_assignment(
        self, mock_mth5_class, mock_read_file, basic_lemi_client, mock_lemi_collection
    ):
        """Test that survey_id and station_id are properly assigned to collection."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_run_ts = MagicMock()
        mock_run_ts.run_metadata.id = "sr1_001"
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        # Replace collection with mock
        basic_lemi_client.collection = mock_lemi_collection

        # Test method
        basic_lemi_client.make_mth5_from_lemi424("custom_survey", "custom_station")

        # Verify collection was configured
        assert basic_lemi_client.collection.survey_id == "custom_survey"
        assert basic_lemi_client.collection.station_id == "custom_station"

    @patch("mth5.clients.lemi424.read_file")
    @patch("mth5.clients.lemi424.MTH5")
    def test_make_mth5_with_calibration_dict(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_lemi_client,
        mock_lemi_collection_with_calibration,
    ):
        """Test MTH5 creation with populated calibration_dict."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_run_ts = MagicMock()
        mock_run_ts.run_metadata.id = "sr1_001"
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        # Replace collection with mock that has calibration_dict
        basic_lemi_client.collection = mock_lemi_collection_with_calibration

        # Test method
        result = basic_lemi_client.make_mth5_from_lemi424("test_survey", "test_station")

        # Verify read_file was called with calibration_dict
        mock_read_file.assert_called_once_with(
            ["file1.txt", "file2.txt"],
            calibration_dict=mock_lemi_collection_with_calibration.calibration_dict,
        )

        assert result == basic_lemi_client.save_path

    @patch("mth5.clients.lemi424.read_file")
    @patch("mth5.clients.lemi424.MTH5")
    def test_make_mth5_calibration_dict_format(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_lemi_client,
        mock_lemi_collection_with_calibration,
        temp_data_dir,
    ):
        """Test that calibration_dict has correct format for magnetic channels."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        mock_run_ts = MagicMock()
        mock_run_ts.run_metadata.id = "sr1_001"
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        # Replace collection with mock that has calibration_dict
        basic_lemi_client.collection = mock_lemi_collection_with_calibration

        # Verify calibration_dict structure
        cal_dict = basic_lemi_client.collection.calibration_dict

        # Should only have magnetic channels
        assert set(cal_dict.keys()) == {"bx", "by", "bz"}

        # Verify file format for each channel
        for channel in ["bx", "by", "bz"]:
            assert channel in cal_dict
            assert cal_dict[channel].endswith(f"lemi-{channel}.sr.json")
            # Verify it's a string path
            assert isinstance(cal_dict[channel], str)

        # Run method to ensure no errors with this format
        basic_lemi_client.make_mth5_from_lemi424("test_survey", "test_station")

        # Verify calibration_dict was passed to read_file
        call_args = mock_read_file.call_args
        assert "calibration_dict" in call_args.kwargs
        assert call_args.kwargs["calibration_dict"] == cal_dict


# =============================================================================
# Property Integration Tests
# =============================================================================


class TestLEMI424ClientPropertyIntegration:
    """Test integration between different properties."""

    def test_h5_kwargs_reflects_property_changes(self, basic_lemi_client):
        """Test that h5_kwargs reflects property changes."""
        # Change properties
        basic_lemi_client.mth5_version = "0.1.0"
        basic_lemi_client.h5_compression = "lzf"
        basic_lemi_client.h5_shuffle = False

        h5_kwargs = basic_lemi_client.h5_kwargs

        assert h5_kwargs["file_version"] == "0.1.0"
        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False

    def test_save_path_components_consistency(self, basic_lemi_client, temp_file_path):
        """Test consistency between save_path components."""
        basic_lemi_client.save_path = temp_file_path

        # All components should be consistent
        assert (
            basic_lemi_client.save_path
            == basic_lemi_client._save_path / basic_lemi_client.mth5_filename
        )
        assert basic_lemi_client.save_path.parent == basic_lemi_client._save_path
        assert basic_lemi_client.save_path.name == basic_lemi_client.mth5_filename


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestLEMI424ClientEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sample_rates_list(self, basic_lemi_client):
        """Test handling of empty sample rates list."""
        basic_lemi_client.sample_rates = []
        assert basic_lemi_client.sample_rates == []

    def test_single_character_string_sample_rate(self, basic_lemi_client):
        """Test single character string sample rate."""
        basic_lemi_client.sample_rates = "5"
        assert basic_lemi_client.sample_rates == [5.0]

    def test_whitespace_in_sample_rates_string(self, basic_lemi_client):
        """Test handling of extra whitespace in sample rates string."""
        basic_lemi_client.sample_rates = " 1 , 2 , 3 "
        assert basic_lemi_client.sample_rates == [1.0, 2.0, 3.0]

    def test_property_attribute_access(self, basic_lemi_client):
        """Test that all expected attributes are accessible."""
        # Test that basic attributes exist and have expected types
        assert hasattr(basic_lemi_client, "data_path")
        assert hasattr(basic_lemi_client, "save_path")
        assert hasattr(basic_lemi_client, "sample_rates")
        assert hasattr(basic_lemi_client, "mth5_filename")
        assert hasattr(basic_lemi_client, "collection")
        assert hasattr(basic_lemi_client, "h5_kwargs")

        assert isinstance(basic_lemi_client.data_path, Path)
        assert isinstance(basic_lemi_client.save_path, Path)
        assert isinstance(basic_lemi_client.sample_rates, list)
        assert isinstance(basic_lemi_client.mth5_filename, str)
        assert isinstance(basic_lemi_client.h5_kwargs, dict)

    def test_logger_attribute(self, basic_lemi_client):
        """Test that logger attribute is properly set."""
        assert hasattr(basic_lemi_client, "logger")
        assert basic_lemi_client.logger is not None

    def test_interact_attribute_default(self, basic_lemi_client):
        """Test that interact attribute has correct default."""
        assert basic_lemi_client.interact is False

    def test_default_hdf5_parameters(self, basic_lemi_client):
        """Test that default HDF5 parameters are correctly set."""
        assert basic_lemi_client.mth5_version == "0.2.0"
        assert basic_lemi_client.h5_compression == "gzip"
        assert basic_lemi_client.h5_compression_opts == 4
        assert basic_lemi_client.h5_shuffle is True
        assert basic_lemi_client.h5_fletcher32 is True
        assert basic_lemi_client.h5_data_level == 1


# =============================================================================
# Inheritance and Base Class Tests
# =============================================================================


class TestLEMI424ClientInheritance:
    """Test inheritance from ClientBase."""

    def test_inherits_from_client_base(self, basic_lemi_client):
        """Test that LEMI424Client properly inherits from ClientBase."""
        from mth5.clients.base import ClientBase

        assert isinstance(basic_lemi_client, ClientBase)

    def test_overrides_collection_initialization(self, basic_lemi_client):
        """Test that LEMI424Client overrides collection initialization."""
        # The collection should be a LEMICollection, not None as in base class
        assert basic_lemi_client.collection is not None
        assert isinstance(basic_lemi_client.collection, LEMICollection)

    def test_default_mth5_filename_override(self, temp_data_dir):
        """Test that default MTH5 filename is overridden."""
        client = LEMI424Client(temp_data_dir)
        assert client.mth5_filename == "from_lemi424.h5"
        # This is different from the base class default of "from_client.h5"


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


class TestLEMI424ClientPerformance:
    """Test performance characteristics."""

    def test_large_sample_rates_list(self, basic_lemi_client):
        """Test handling of large sample rates list."""
        large_sample_rates = list(range(1, 101))  # 100 sample rates
        basic_lemi_client.sample_rates = large_sample_rates
        assert len(basic_lemi_client.sample_rates) == 100
        assert basic_lemi_client.sample_rates == [float(x) for x in large_sample_rates]

    def test_very_long_filename(self, basic_lemi_client, temp_data_dir):
        """Test handling of very long filenames."""
        # Use a shorter repetition to avoid platform-dependent filename length limits
        long_filename = "very_" * 10 + "long_filename.h5"
        long_path = temp_data_dir / long_filename

        basic_lemi_client.save_path = long_path
        assert basic_lemi_client.mth5_filename == long_filename

    def test_nested_directory_creation(self, basic_lemi_client, temp_data_dir):
        """Test creation of nested directories (limited by base class implementation)."""
        # Create the intermediate directories first since base class doesn't use parents=True
        level1_dir = temp_data_dir / "level1"
        level1_dir.mkdir()
        level2_dir = level1_dir / "level2"
        level2_dir.mkdir()

        nested_path = level2_dir / "level3" / "test.h5"

        basic_lemi_client.save_path = nested_path

        assert nested_path.parent.exists()
        assert basic_lemi_client.save_path == nested_path


# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])
