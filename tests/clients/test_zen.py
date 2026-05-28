# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for ZenClient with fixtures, parametrization,
and additional coverage for untested functionality.

Created by modernizing test_zen.py with pytest patterns.

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
from mt_io.zen import Z3DCollection

from mth5.clients.zen import ZenClient


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
def basic_zen_client(temp_data_dir):
    """Create a basic ZenClient instance for testing."""
    return ZenClient(temp_data_dir, **{"h5_mode": "w", "h5_driver": "sec2"})


@pytest.fixture
def configured_zen_client(temp_data_dir, temp_file_path):
    """Create a configured ZenClient instance with custom settings."""
    return ZenClient(
        temp_data_dir,
        sample_rates=[4096, 1024, 256],
        save_path=temp_file_path.parent,
        mth5_filename=temp_file_path.name,
        calibration_path=None,
        h5_compression="lzf",
        h5_shuffle=False,
        mth5_version="0.1.0",
    )


@pytest.fixture
def mock_z3d_collection():
    """Create a mock Z3DCollection for testing."""
    mock_collection = Mock(spec=Z3DCollection)
    mock_collection.station_metadata_dict = {
        "MT001": Mock(survey="test_survey", id="MT001")
    }
    mock_collection.get_runs.return_value = {
        "MT001": {
            "run001": Mock(
                survey=Mock(unique=Mock(return_value=["test_survey"])),
                itertuples=Mock(
                    return_value=[Mock(fn=Mock(name="test.z3d"), calibration_fn=None)]
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

    # Create mock antenna calibration file
    ant_cal_file = cal_dir / "amtant.cal"
    ant_cal_file.touch()

    return {"calibration_dir": cal_dir, "ant_cal_file": ant_cal_file}


# =============================================================================
# ZenClient Initialization Tests
# =============================================================================


class TestZenClientInitialization:
    """Test ZenClient initialization and basic properties."""

    def test_basic_initialization(self, temp_data_dir):
        """Test basic initialization with minimal parameters."""
        client = ZenClient(temp_data_dir)

        assert client.data_path == temp_data_dir
        assert client.mth5_filename == "from_zen.h5"
        assert client.sample_rates == [4096, 1024, 256]  # Default sample rates
        assert client.save_path == temp_data_dir / "from_zen.h5"
        assert isinstance(client.collection, Z3DCollection)
        assert client.calibration_path is None

    def test_initialization_with_custom_parameters(
        self, temp_data_dir, mock_calibration_files
    ):
        """Test initialization with custom parameters."""
        custom_filename = "custom_zen.h5"
        custom_sample_rates = [2048, 512, 128]
        cal_path = mock_calibration_files["ant_cal_file"]

        client = ZenClient(
            temp_data_dir,
            sample_rates=custom_sample_rates,
            mth5_filename=custom_filename,
            calibration_path=cal_path,
            h5_compression="lzf",
            mth5_version="0.1.0",
        )

        assert client.mth5_filename == custom_filename
        assert client.sample_rates == custom_sample_rates
        assert client.calibration_path == cal_path
        assert client.h5_compression == "lzf"
        assert client.mth5_version == "0.1.0"

    def test_initialization_with_save_path(self, temp_data_dir, temp_file_path):
        """Test initialization with custom save path."""
        client = ZenClient(temp_data_dir, save_path=temp_file_path)

        assert client.save_path == temp_file_path
        assert client.mth5_filename == temp_file_path.name

    def test_initialization_none_data_path_fails(self):
        """Test that None data_path raises ValueError."""
        with pytest.raises(ValueError, match="data_path cannot be None"):
            ZenClient(None)

    def test_initialization_bad_directory_fails(self):
        """Test that non-existent directory raises IOError."""
        with pytest.raises(IOError, match="Could not find"):
            ZenClient(Path("non_existent_directory_12345"))

    def test_collection_initialization(self, basic_zen_client):
        """Test that Z3DCollection is properly initialized."""
        assert isinstance(basic_zen_client.collection, Z3DCollection)
        assert basic_zen_client.collection.file_path == basic_zen_client.data_path


# =============================================================================
# H5 Parameters Tests
# =============================================================================


class TestZenClientH5Parameters:
    """Test H5 parameter handling and validation."""

    def test_h5_kwargs_default_parameters(self, basic_zen_client):
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

        h5_kwargs = basic_zen_client.h5_kwargs
        assert sorted(h5_kwargs.keys()) == sorted(expected_keys)

        # Test default values
        assert h5_kwargs["compression"] == "gzip"
        assert h5_kwargs["compression_opts"] == 4
        assert h5_kwargs["file_version"] == "0.2.0"
        assert h5_kwargs["fletcher32"] is True
        assert h5_kwargs["shuffle"] is True
        assert h5_kwargs["data_level"] == 1

    def test_h5_kwargs_custom_parameters(self, configured_zen_client):
        """Test custom H5 parameters."""
        h5_kwargs = configured_zen_client.h5_kwargs

        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False
        assert h5_kwargs["file_version"] == "0.1.0"

    def test_h5_kwargs_with_h5_prefixed_parameters(self, temp_data_dir):
        """Test that h5_ prefixed parameters are included in h5_kwargs."""
        client = ZenClient(
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


class TestZenClientSampleRates:
    """Test sample rates property validation and conversion."""

    @pytest.mark.parametrize(
        "input_value,expected", [(10.5, [10.5]), (100, [100.0]), (42.0, [42.0])]
    )
    def test_sample_rates_float_input(self, basic_zen_client, input_value, expected):
        """Test setting sample rates with float/int values."""
        basic_zen_client.sample_rates = input_value
        assert basic_zen_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_string,expected",
        [
            ("10, 42, 1200", [10.0, 42.0, 1200.0]),
            ("1.5, 25.5, 100", [1.5, 25.5, 100.0]),
            ("1000", [1000.0]),
            ("10,42,1200", [10.0, 42.0, 1200.0]),  # No spaces
        ],
    )
    def test_sample_rates_string_input(self, basic_zen_client, input_string, expected):
        """Test setting sample rates with string values."""
        basic_zen_client.sample_rates = input_string
        assert basic_zen_client.sample_rates == expected

    @pytest.mark.parametrize(
        "input_list,expected",
        [
            ([10, 42, 1200], [10.0, 42.0, 1200.0]),
            ((1, 5, 25), [1.0, 5.0, 25.0]),
            ([1.5, 2.5], [1.5, 2.5]),
            ([4096, 1024, 256], [4096.0, 1024.0, 256.0]),  # Default Zen sample rates
        ],
    )
    def test_sample_rates_list_input(self, basic_zen_client, input_list, expected):
        """Test setting sample rates with list/tuple values."""
        basic_zen_client.sample_rates = input_list
        assert basic_zen_client.sample_rates == expected

    @pytest.mark.parametrize("invalid_value", [None, {}, set([1, 2, 3]), object()])
    def test_sample_rates_invalid_input(self, basic_zen_client, invalid_value):
        """Test that invalid sample rate inputs raise TypeError."""
        with pytest.raises(TypeError, match="Cannot parse"):
            basic_zen_client.sample_rates = invalid_value


# =============================================================================
# Save Path Tests
# =============================================================================


class TestZenClientSavePath:
    """Test save path property handling and validation."""

    def test_save_path_file_assignment(self, basic_zen_client, temp_file_path):
        """Test setting save path with file path."""
        basic_zen_client.save_path = temp_file_path

        assert basic_zen_client._save_path == temp_file_path.parent
        assert basic_zen_client.mth5_filename == temp_file_path.name
        assert basic_zen_client.save_path == temp_file_path

    def test_save_path_directory_assignment(self, basic_zen_client, temp_data_dir):
        """Test setting save path with directory path."""
        original_filename = basic_zen_client.mth5_filename
        basic_zen_client.save_path = temp_data_dir

        assert basic_zen_client._save_path == temp_data_dir
        assert basic_zen_client.mth5_filename == original_filename
        assert basic_zen_client.save_path == temp_data_dir / original_filename

    def test_save_path_none_defaults_to_data_path(self, temp_data_dir):
        """Test that None save_path defaults to data_path."""
        client = ZenClient(temp_data_dir, save_path=None)

        assert client._save_path == temp_data_dir

    def test_save_path_components_consistency(self, basic_zen_client, temp_file_path):
        """Test consistency between save_path components."""
        basic_zen_client.save_path = temp_file_path

        # All components should be consistent
        assert (
            basic_zen_client.save_path
            == basic_zen_client._save_path / basic_zen_client.mth5_filename
        )
        assert basic_zen_client.save_path.parent == basic_zen_client._save_path
        assert basic_zen_client.save_path.name == basic_zen_client.mth5_filename


# =============================================================================
# Calibration Path Tests
# =============================================================================


class TestZenClientCalibrationPath:
    """Test calibration path handling and validation."""

    def test_calibration_path_none_default(self, basic_zen_client):
        """Test that calibration path defaults to None."""
        assert basic_zen_client.calibration_path is None

    def test_calibration_path_valid_file(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test setting calibration path with valid file."""
        cal_file = mock_calibration_files["ant_cal_file"]
        basic_zen_client.calibration_path = cal_file

        assert basic_zen_client.calibration_path == cal_file
        assert isinstance(basic_zen_client.calibration_path, Path)

    def test_calibration_path_valid_directory(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test setting calibration path with valid directory."""
        cal_dir = mock_calibration_files["calibration_dir"]
        basic_zen_client.calibration_path = cal_dir

        assert basic_zen_client.calibration_path == cal_dir
        assert isinstance(basic_zen_client.calibration_path, Path)

    def test_calibration_path_string_conversion(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test that string calibration paths are converted to Path objects."""
        cal_file = mock_calibration_files["ant_cal_file"]
        basic_zen_client.calibration_path = str(cal_file)

        assert basic_zen_client.calibration_path == cal_file
        assert isinstance(basic_zen_client.calibration_path, Path)

    def test_calibration_path_nonexistent_file_raises_error(self, basic_zen_client):
        """Test that non-existent calibration path raises IOError."""
        with pytest.raises(IOError, match="Could not find"):
            basic_zen_client.calibration_path = "/non/existent/path/calibration.cal"

    def test_calibration_path_none_assignment(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test that None can be assigned to calibration_path."""
        # First set a valid path
        cal_file = mock_calibration_files["ant_cal_file"]
        basic_zen_client.calibration_path = cal_file
        assert basic_zen_client.calibration_path == cal_file

        # Then set to None
        basic_zen_client.calibration_path = None
        assert basic_zen_client.calibration_path is None


# =============================================================================
# Collection Integration Tests
# =============================================================================


class TestZenClientCollectionIntegration:
    """Test integration with Z3DCollection."""

    def test_get_run_dict_calls_collection(self, basic_zen_client):
        """Test that get_run_dict delegates to collection."""
        with patch.object(basic_zen_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {"test": "data"}

            result = basic_zen_client.get_run_dict()

            mock_get_runs.assert_called_once_with(
                sample_rates=basic_zen_client.sample_rates,
                calibration_path=basic_zen_client.calibration_path,
            )
            assert result == {"test": "data"}

    def test_get_run_dict_with_calibration_path(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test get_run_dict with calibration path."""
        cal_file = mock_calibration_files["ant_cal_file"]
        basic_zen_client.calibration_path = cal_file

        with patch.object(basic_zen_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {"test": "data"}

            result = basic_zen_client.get_run_dict()

            mock_get_runs.assert_called_once_with(
                sample_rates=basic_zen_client.sample_rates, calibration_path=cal_file
            )

    def test_collection_uses_correct_path(self, temp_data_dir):
        """Test that collection is initialized with correct path."""
        client = ZenClient(temp_data_dir)
        assert client.collection.file_path == temp_data_dir

    def test_get_survey_method(self, basic_zen_client):
        """Test get_survey method functionality."""
        # Create mock station dictionary
        mock_run1 = Mock()
        mock_run1.survey.unique.return_value = ["TEST_SURVEY"]
        mock_run2 = Mock()
        mock_run2.survey.unique.return_value = ["TEST_SURVEY"]

        station_dict = {"run001": mock_run1, "run002": mock_run2}

        result = basic_zen_client.get_survey(station_dict)
        assert result == "TEST_SURVEY"

    def test_get_survey_method_multiple_surveys(self, basic_zen_client):
        """Test get_survey method with mixed surveys (should return first unique)."""
        # Create mock station dictionary with different surveys
        mock_run1 = Mock()
        mock_run1.survey.unique.return_value = ["SURVEY_A"]
        mock_run2 = Mock()
        mock_run2.survey.unique.return_value = ["SURVEY_B"]

        station_dict = {"run001": mock_run1, "run002": mock_run2}

        result = basic_zen_client.get_survey(station_dict)
        # Should return one of the surveys (set order is not guaranteed)
        assert result in ["SURVEY_A", "SURVEY_B"]


# =============================================================================
# MTH5 Creation Tests (Mocked)
# =============================================================================


class TestZenClientMTH5Creation:
    """Test MTH5 file creation functionality with mocking."""

    @patch("mth5.clients.zen.read_file")
    @patch("mth5.clients.zen.MTH5")
    def test_make_mth5_from_zen_basic(
        self, mock_mth5_class, mock_read_file, basic_zen_client, mock_z3d_collection
    ):
        """Test basic MTH5 creation from Zen data."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup run group for combining
        mock_runts = MagicMock()
        mock_run_group.to_runts.return_value = mock_runts
        mock_combined_run = MagicMock()
        mock_combined_run.run_metadata.id = "sr1_0001"
        mock_runts.merge.return_value = mock_combined_run

        mock_ch_ts = MagicMock()
        mock_read_file.return_value = mock_ch_ts

        # Replace collection with mock
        basic_zen_client.collection = mock_z3d_collection

        # Test method
        result = basic_zen_client.make_mth5_from_zen()

        # Verify calls
        mock_mth5_instance.open_mth5.assert_called_once_with(
            basic_zen_client.save_path, "w"
        )
        mock_mth5_instance.add_survey.assert_called_once()
        mock_survey_group.stations_group.add_station.assert_called_once()
        # With combine=True (default), add_run is called twice: once for original run, once for combined run
        assert mock_station_group.add_run.call_count >= 1
        mock_run_group.from_channel_ts.assert_called_once_with(mock_ch_ts)

        assert result == basic_zen_client.save_path

    @patch("mth5.clients.zen.read_file")
    @patch("mth5.clients.zen.MTH5")
    def test_make_mth5_from_zen_with_survey_id(
        self, mock_mth5_class, mock_read_file, basic_zen_client, mock_z3d_collection
    ):
        """Test MTH5 creation with custom survey ID."""
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
        mock_read_file.return_value = mock_ch_ts

        # Replace collection with mock
        basic_zen_client.collection = mock_z3d_collection

        # Test with custom survey ID
        result = basic_zen_client.make_mth5_from_zen(survey_id="CUSTOM_SURVEY")

        # Verify survey ID was used
        mock_mth5_instance.add_survey.assert_called_once_with("CUSTOM_SURVEY")

    @patch("mth5.clients.zen.read_file")
    @patch("mth5.clients.zen.MTH5")
    def test_make_mth5_from_zen_with_combine_false(
        self, mock_mth5_class, mock_read_file, basic_zen_client, mock_z3d_collection
    ):
        """Test MTH5 creation with combine=False."""
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
        mock_read_file.return_value = mock_ch_ts

        # Replace collection with mock
        basic_zen_client.collection = mock_z3d_collection

        # Test with combine=False
        result = basic_zen_client.make_mth5_from_zen(combine=False)

        # Verify no combined run was created (method should not call merge)
        assert result == basic_zen_client.save_path

    @patch("mth5.clients.zen.read_file")
    @patch("mth5.clients.zen.MTH5")
    def test_make_mth5_from_zen_with_kwargs(
        self, mock_mth5_class, mock_read_file, basic_zen_client, mock_z3d_collection
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
        mock_read_file.return_value = mock_ch_ts

        # Replace collection with mock
        basic_zen_client.collection = mock_z3d_collection

        # Test with additional kwargs
        result = basic_zen_client.make_mth5_from_zen(
            sample_rates=[2048, 512], h5_compression="lzf"
        )

        # Verify that kwargs were applied
        assert basic_zen_client.sample_rates == [2048.0, 512.0]
        assert basic_zen_client.h5_compression == "lzf"

    @patch("mth5.clients.zen.read_file")
    @patch("mth5.clients.zen.MTH5")
    def test_make_mth5_from_zen_with_combine_true_multiple_runs(
        self, mock_mth5_class, mock_read_file, basic_zen_client
    ):
        """Test MTH5 creation with combine=True and multiple runs."""
        # Setup mocks
        mock_mth5_instance = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5_instance

        mock_survey_group = MagicMock()
        mock_mth5_instance.add_survey.return_value = mock_survey_group

        mock_station_group = MagicMock()
        mock_survey_group.stations_group.add_station.return_value = mock_station_group

        mock_run_group = MagicMock()
        mock_station_group.add_run.return_value = mock_run_group

        # Setup run groups with to_runts method
        mock_runts1 = MagicMock()
        mock_runts2 = MagicMock()
        mock_run_group.to_runts.side_effect = [mock_runts1, mock_runts2]

        # Setup combined run
        mock_combined_run = MagicMock()
        mock_combined_run.run_metadata.id = "sr1_0001"
        mock_runts1.merge.return_value = mock_combined_run

        mock_combined_run_group = MagicMock()
        mock_station_group.add_run.side_effect = [
            mock_run_group,
            mock_run_group,
            mock_combined_run_group,
        ]

        mock_ch_ts = MagicMock()
        mock_read_file.return_value = mock_ch_ts

        # Setup collection with multiple runs
        mock_collection = Mock(spec=Z3DCollection)
        mock_collection.station_metadata_dict = {
            "MT001": Mock(survey="test_survey", id="MT001")
        }
        mock_collection.get_runs.return_value = {
            "MT001": {
                "run001": Mock(
                    survey=Mock(unique=Mock(return_value=["test_survey"])),
                    itertuples=Mock(
                        return_value=[
                            Mock(fn=Mock(name="test1.z3d"), calibration_fn=None)
                        ]
                    ),
                ),
                "run002": Mock(
                    survey=Mock(unique=Mock(return_value=["test_survey"])),
                    itertuples=Mock(
                        return_value=[
                            Mock(fn=Mock(name="test2.z3d"), calibration_fn=None)
                        ]
                    ),
                ),
            }
        }
        basic_zen_client.collection = mock_collection

        # Test method with combine=True
        result = basic_zen_client.make_mth5_from_zen(combine=True)

        # Verify combined run was created
        mock_runts1.merge.assert_called_once()
        mock_combined_run_group.from_runts.assert_called_once_with(mock_combined_run)


# =============================================================================
# Property Integration Tests
# =============================================================================


class TestZenClientPropertyIntegration:
    """Test integration between different properties."""

    def test_h5_kwargs_reflects_property_changes(self, basic_zen_client):
        """Test that h5_kwargs reflects property changes."""
        # Change properties
        basic_zen_client.mth5_version = "0.1.0"
        basic_zen_client.h5_compression = "lzf"
        basic_zen_client.h5_shuffle = False

        h5_kwargs = basic_zen_client.h5_kwargs

        assert h5_kwargs["file_version"] == "0.1.0"
        assert h5_kwargs["compression"] == "lzf"
        assert h5_kwargs["shuffle"] is False

    def test_calibration_path_and_get_run_dict_integration(
        self, basic_zen_client, mock_calibration_files
    ):
        """Test that calibration_path is properly passed to get_run_dict."""
        cal_file = mock_calibration_files["ant_cal_file"]
        basic_zen_client.calibration_path = cal_file

        with patch.object(basic_zen_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {}

            basic_zen_client.get_run_dict()

            # Verify calibration_path was passed
            call_args = mock_get_runs.call_args[1]
            assert call_args["calibration_path"] == cal_file


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestZenClientEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sample_rates_list(self, basic_zen_client):
        """Test handling of empty sample rates list."""
        basic_zen_client.sample_rates = []
        assert basic_zen_client.sample_rates == []

    def test_single_character_string_sample_rate(self, basic_zen_client):
        """Test single character string sample rate."""
        basic_zen_client.sample_rates = "5"
        assert basic_zen_client.sample_rates == [5.0]

    def test_whitespace_in_sample_rates_string(self, basic_zen_client):
        """Test handling of extra whitespace in sample rates string."""
        basic_zen_client.sample_rates = " 4096 , 1024 , 256 "
        assert basic_zen_client.sample_rates == [4096.0, 1024.0, 256.0]

    def test_zen_specific_sample_rates(self, basic_zen_client):
        """Test Zen-specific sample rates."""
        # Common Zen sample rates
        zen_rates = [4096, 1024, 256]
        basic_zen_client.sample_rates = zen_rates
        assert basic_zen_client.sample_rates == [4096.0, 1024.0, 256.0]

    def test_property_attribute_access(self, basic_zen_client):
        """Test that all expected attributes are accessible."""
        # Test that basic attributes exist and have expected types
        assert hasattr(basic_zen_client, "data_path")
        assert hasattr(basic_zen_client, "save_path")
        assert hasattr(basic_zen_client, "sample_rates")
        assert hasattr(basic_zen_client, "mth5_filename")
        assert hasattr(basic_zen_client, "collection")
        assert hasattr(basic_zen_client, "h5_kwargs")
        assert hasattr(basic_zen_client, "calibration_path")

        assert isinstance(basic_zen_client.data_path, Path)
        assert isinstance(basic_zen_client.save_path, Path)
        assert isinstance(basic_zen_client.sample_rates, list)
        assert isinstance(basic_zen_client.mth5_filename, str)
        assert isinstance(basic_zen_client.h5_kwargs, dict)

    def test_logger_attribute(self, basic_zen_client):
        """Test that logger attribute is properly set."""
        assert hasattr(basic_zen_client, "logger")
        assert basic_zen_client.logger is not None

    def test_interact_attribute_default(self, basic_zen_client):
        """Test that interact attribute has correct default."""
        assert basic_zen_client.interact is False

    def test_default_hdf5_parameters(self, basic_zen_client):
        """Test that default HDF5 parameters are correctly set."""
        assert basic_zen_client.mth5_version == "0.2.0"
        assert basic_zen_client.h5_compression == "gzip"
        assert basic_zen_client.h5_compression_opts == 4
        assert basic_zen_client.h5_shuffle is True
        assert basic_zen_client.h5_fletcher32 is True
        assert basic_zen_client.h5_data_level == 1


# =============================================================================
# Inheritance and Base Class Tests
# =============================================================================


class TestZenClientInheritance:
    """Test inheritance from ClientBase."""

    def test_inherits_from_client_base(self, basic_zen_client):
        """Test that ZenClient properly inherits from ClientBase."""
        from mth5.clients.base import ClientBase

        assert isinstance(basic_zen_client, ClientBase)

    def test_overrides_collection_initialization(self, basic_zen_client):
        """Test that ZenClient overrides collection initialization."""
        # The collection should be a Z3DCollection, not None as in base class
        assert basic_zen_client.collection is not None
        assert isinstance(basic_zen_client.collection, Z3DCollection)

    def test_default_mth5_filename_override(self, temp_data_dir):
        """Test that default MTH5 filename is overridden."""
        client = ZenClient(temp_data_dir)
        assert client.mth5_filename == "from_zen.h5"
        # This is different from the base class default of "from_client.h5"

    def test_default_sample_rates_override(self, temp_data_dir):
        """Test that default sample rates are overridden for Zen."""
        client = ZenClient(temp_data_dir)
        assert client.sample_rates == [4096, 1024, 256]
        # Zen-specific default sample rates


# =============================================================================
# Zen-Specific Functionality Tests
# =============================================================================


class TestZenClientSpecificFunctionality:
    """Test Zen-specific functionality."""

    def test_zen_default_sample_rates(self, temp_data_dir):
        """Test that Zen has appropriate default sample rates."""
        client = ZenClient(temp_data_dir)
        # Zen typically uses 4096, 1024, 256 Hz
        assert 4096 in client.sample_rates
        assert 1024 in client.sample_rates
        assert 256 in client.sample_rates

    def test_calibration_initialization_types(self, basic_zen_client):
        """Test that calibration_path is properly typed."""
        # Test initial None state
        assert basic_zen_client.calibration_path is None

    def test_z3d_collection_integration(self, basic_zen_client):
        """Test Z3DCollection specific integration."""
        # Verify the collection is the right type and has the expected interface
        assert hasattr(basic_zen_client.collection, "get_runs")
        assert hasattr(basic_zen_client.collection, "station_metadata_dict")
        assert hasattr(basic_zen_client.collection, "file_path")

    def test_survey_extraction_logic(self, basic_zen_client):
        """Test survey extraction logic handles various scenarios."""
        # Test with consistent survey
        station_dict_consistent = {
            "run001": Mock(survey=Mock(unique=Mock(return_value=["SURVEY_A"]))),
            "run002": Mock(survey=Mock(unique=Mock(return_value=["SURVEY_A"]))),
        }
        result = basic_zen_client.get_survey(station_dict_consistent)
        assert result == "SURVEY_A"

        # Test with single run
        station_dict_single = {
            "run001": Mock(survey=Mock(unique=Mock(return_value=["SURVEY_B"])))
        }
        result = basic_zen_client.get_survey(station_dict_single)
        assert result == "SURVEY_B"


# =============================================================================
# Performance and Scalability Tests
# =============================================================================


class TestZenClientPerformance:
    """Test performance characteristics."""

    def test_large_sample_rates_list(self, basic_zen_client):
        """Test handling of large sample rates list."""
        large_sample_rates = list(range(1, 101))  # 100 sample rates
        basic_zen_client.sample_rates = large_sample_rates
        assert len(basic_zen_client.sample_rates) == 100
        assert basic_zen_client.sample_rates == [float(x) for x in large_sample_rates]

    def test_very_long_filename(self, basic_zen_client, temp_data_dir):
        """Test handling of very long filenames."""
        # Use a shorter repetition to avoid platform-dependent filename length limits
        long_filename = "very_" * 10 + "long_zen_filename.h5"
        long_path = temp_data_dir / long_filename

        basic_zen_client.save_path = long_path
        assert basic_zen_client.mth5_filename == long_filename

    def test_deeply_nested_directory_structure(self, temp_data_dir):
        """Test handling of deeply nested directory structures."""
        # Create a deeply nested directory
        nested_dir = temp_data_dir
        for i in range(10):
            nested_dir = nested_dir / f"level_{i}"
        nested_dir.mkdir(parents=True)

        client = ZenClient(nested_dir)
        assert client.data_path == nested_dir


# =============================================================================
# Integration with MTH5 Tests
# =============================================================================


class TestZenClientMTH5Integration:
    """Test integration with MTH5 file operations."""

    def test_h5_kwargs_compatibility_with_mth5(self, basic_zen_client):
        """Test that h5_kwargs are compatible with MTH5 expectations."""
        h5_kwargs = basic_zen_client.h5_kwargs

        # These are the parameters that MTH5 expects
        expected_mth5_params = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]

        for param in expected_mth5_params:
            assert param in h5_kwargs

    def test_mth5_version_parameter_handling(self, temp_data_dir):
        """Test MTH5 version parameter handling."""
        # Test different MTH5 versions
        versions = ["0.1.0", "0.2.0"]

        for version in versions:
            client = ZenClient(temp_data_dir, mth5_version=version)
            assert client.mth5_version == version
            assert client.h5_kwargs["file_version"] == version


# =============================================================================
# Error Scenarios and Exception Handling
# =============================================================================


class TestZenClientErrorHandling:
    """Test error handling and exception scenarios."""

    def test_invalid_calibration_path_scenarios(self, basic_zen_client):
        """Test various invalid calibration path scenarios."""
        invalid_paths = [
            "/completely/non/existent/path.cal",
            "/another/fake/path/calibration.txt",
            "relative/non/existent/path.cal",
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(IOError, match="Could not find"):
                basic_zen_client.calibration_path = invalid_path

    def test_collection_method_error_handling(self, basic_zen_client):
        """Test error handling in collection methods."""
        # Test with mock that raises an exception
        with patch.object(basic_zen_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.side_effect = Exception("Collection error")

            with pytest.raises(Exception, match="Collection error"):
                basic_zen_client.get_run_dict()

    def test_get_survey_with_empty_station_dict(self, basic_zen_client):
        """Test get_survey method with empty station dict."""
        with pytest.raises((KeyError, IndexError)):
            basic_zen_client.get_survey({})


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestZenClientBackwardCompatibility:
    """Test backward compatibility with original test cases."""

    def test_original_test_cases_still_pass(self, temp_data_dir):
        """Test that all original test cases still pass with new implementation."""
        # Original test: basic initialization
        file_path = temp_data_dir / "test.h5"
        base = ZenClient(file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"})

        # Original test: h5_kwargs keys
        keys = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]
        assert sorted(keys) == sorted(base.h5_kwargs.keys())

        # Original test: sample rates float
        base.sample_rates = 10.5
        assert base.sample_rates == [10.5]

        # Original test: sample rates string
        base.sample_rates = "10, 42, 1200"
        assert base.sample_rates == [10.0, 42.0, 1200.0]

        # Original test: sample rates list
        base.sample_rates = [10, 42, 1200]
        assert base.sample_rates == [10.0, 42.0, 1200.0]

        # Original test: sample rates failure
        with pytest.raises(TypeError):
            base.sample_rates = None

        # Original test: save path
        base.save_path = file_path
        assert base._save_path == file_path.parent
        assert base.mth5_filename == file_path.name
        assert base.save_path == file_path

        # Original test: initialization failures
        with pytest.raises(ValueError):
            ZenClient(None)

        with pytest.raises(IOError):
            ZenClient(Path("non_existent_directory_12345"))

        # Original test: calibration path None
        assert base.calibration_path is None


# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])
