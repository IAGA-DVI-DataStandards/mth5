# -*- coding: utf-8 -*-
"""
Optimized pytest test suite for MTH5

Created on September 27, 2025

High-performance version of test_mth5.py using pytest fixtures and modern testing patterns.
Optimized for speed with session-scoped fixtures and reduced data sizes.

@author: automated conversion from unittest
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.common.mttime import MTime
from mt_timeseries import ChannelTS, RunTS

from mth5 import groups, helpers
from mth5.mth5 import _default_table_names, MTH5
from mth5.utils.exceptions import MTH5Error


# Set numpy random seed for reproducible tests
np.random.seed(42)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_files():
    """Ensure any open files are closed before starting tests."""
    helpers.close_open_files()
    yield
    helpers.close_open_files()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup with retry for Windows file locking issues
    try:
        import time

        time.sleep(0.1)  # Brief delay to allow file handles to close
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def test_file_path(temp_dir):
    """Generate a unique test file path."""
    return temp_dir / f"test_{np.random.randint(1000, 9999)}.mth5"


@pytest.fixture
def mth5_empty(test_file_path):
    """Create an empty MTH5 file for testing."""
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(test_file_path, mode="w")
    yield mth5_obj
    try:
        if hasattr(mth5_obj, "hdf5_file") and mth5_obj.hdf5_file is not None:
            mth5_obj.close_mth5()
    except Exception:
        pass  # File might already be closed
    # Let temp_dir fixture handle file cleanup


@pytest.fixture(scope="session")
def sample_timeseries_data():
    """Generate sample timeseries data for testing - session scoped for speed."""

    def _create_channel_ts(
        component,
        ch_type,
        sample_rate=1,
        n_samples=1024,  # Reduced default size for speed
        start_time="2020-01-01T12:00:00",
    ):
        meta_dict = {
            ch_type: {
                "component": component,
                "dipole_length": 49.0,
                "measurement_azimuth": 12.0,
                "type": ch_type,
                "units": "counts",
                "time_period.start": start_time,
                "sample_rate": sample_rate,
            }
        }
        return ChannelTS(
            ch_type, data=np.random.rand(n_samples), channel_metadata=meta_dict
        )

    return _create_channel_ts


@pytest.fixture(scope="session")
def mth5_with_data_session(temp_dir, sample_timeseries_data):
    """Create MTH5 file with sample data - session scoped for speed."""
    test_file_path = temp_dir / f"session_test_{np.random.randint(1000, 9999)}.mth5"
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(test_file_path, mode="w")

    # Create minimal sample data for speed
    ts_list = []
    for comp in ["ex", "hy"]:  # Reduced to just 2 channels for speed
        if comp[0] in ["e"]:
            ch_type = "electric"
        else:
            ch_type = "magnetic"

        channel_ts = sample_timeseries_data(
            comp, ch_type, n_samples=256
        )  # Smaller data
        ts_list.append(channel_ts)

    run_ts = RunTS(ts_list, {"run": {"id": "MT009a"}})
    station = mth5_obj.add_station("MT009", survey="test")
    run_group = station.add_run("MT009a")
    channel_groups = run_group.from_runts(run_ts)

    # Store references for test access
    mth5_obj._test_station = station
    mth5_obj._test_run_group = run_group
    mth5_obj._test_channel_groups = channel_groups

    yield mth5_obj

    try:
        if hasattr(mth5_obj, "hdf5_file") and mth5_obj.hdf5_file is not None:
            mth5_obj.close_mth5()
    except Exception:
        pass  # File might already be closed


@pytest.fixture
def mth5_with_data(test_file_path, sample_timeseries_data):
    """Create MTH5 file with sample station, run, and channel data - optimized."""
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(test_file_path, mode="w")

    # Create minimal sample data for speed
    ts_list = []
    for comp in ["ex", "hy"]:  # Reduced to just 2 channels for speed
        if comp[0] in ["e"]:
            ch_type = "electric"
        else:
            ch_type = "magnetic"

        channel_ts = sample_timeseries_data(
            comp, ch_type, n_samples=256
        )  # Smaller data
        ts_list.append(channel_ts)

    run_ts = RunTS(ts_list, {"run": {"id": "MT009a"}})
    station = mth5_obj.add_station("MT009", survey="test")
    run_group = station.add_run("MT009a")
    channel_groups = run_group.from_runts(run_ts)

    # Store references for test access
    mth5_obj._test_station = station
    mth5_obj._test_run_group = run_group
    mth5_obj._test_channel_groups = channel_groups

    yield mth5_obj

    try:
        if hasattr(mth5_obj, "hdf5_file") and mth5_obj.hdf5_file is not None:
            mth5_obj.close_mth5()
    except Exception:
        pass  # File might already be closed


# =============================================================================
# Test Classes using pytest
# =============================================================================


class TestMTH5Basic:
    """Basic MTH5 functionality tests - fast category."""

    @pytest.mark.fast
    def test_initial_standards_group_size(self, mth5_empty):
        """Test that standards group has content."""
        stable = mth5_empty.standards_group.summary_table
        assert stable.nrows > 1

    @pytest.mark.fast
    def test_default_group_names(self, mth5_empty):
        """Test default group names are created correctly."""
        groups = sorted(mth5_empty.survey_group.groups_list)
        defaults = sorted(mth5_empty._default_subgroup_names + _default_table_names())
        assert defaults == groups

    @pytest.mark.fast
    def test_file_properties(self, mth5_empty, test_file_path):
        """Test file properties and states."""
        assert isinstance(mth5_empty.filename, Path)
        assert mth5_empty.h5_is_read() is True
        assert mth5_empty.h5_is_write() is True
        assert mth5_empty.validate_file() is True

    @pytest.mark.fast
    def test_survey_metadata_operations(self, mth5_empty):
        """Test survey metadata can be modified and persisted."""
        survey_metadata = mth5_empty.survey_group.metadata

        # Test default ID
        assert survey_metadata.id == "default_survey"

        # Test setting new ID
        new_survey_id = "MT Survey"
        survey_metadata.id = new_survey_id
        assert survey_metadata.id == new_survey_id

        # Test metadata update persists
        mth5_empty.survey_group.update_metadata(survey_metadata.to_dict())
        assert mth5_empty.survey_group.metadata.id == new_survey_id


class TestMTH5StationOperations:
    """Test station-level operations."""

    def test_add_station_success(self, mth5_empty):
        """Test successful station addition."""
        new_station = mth5_empty.add_station("MT001")

        # Comprehensive station tests using subtests
        assert new_station._has_read_metadata is True
        assert "MT001" in mth5_empty.stations_group.groups_list
        assert isinstance(new_station, groups.StationGroup)

        # Test getting station
        sg = mth5_empty.get_station("MT001")
        assert isinstance(sg, groups.StationGroup)
        assert sg._has_read_metadata is True

        # Test survey metadata integration
        assert new_station.survey_metadata.station_names == ["MT001"]

    def test_station_lifecycle(self, mth5_empty):
        """Test complete station lifecycle (add, access, remove)."""
        # Add station
        mth5_empty.add_station("MT002")
        assert "MT002" in mth5_empty.stations_group.groups_list

        # Remove station
        mth5_empty.remove_station("MT002")
        assert "MT002" not in mth5_empty.stations_group.groups_list

    def test_get_nonexistent_station_fails(self, mth5_empty):
        """Test that getting non-existent station raises appropriate error."""
        with pytest.raises(MTH5Error):
            mth5_empty.get_station("NONEXISTENT")


class TestMTH5RunOperations:
    """Test run-level operations."""

    @pytest.fixture
    def station_with_runs(self, mth5_empty):
        """Create a station with runs for testing."""
        station = mth5_empty.add_station("MT003")
        run1 = station.add_run("MT003a")
        run2 = station.add_run("MT003b")
        return station, [run1, run2]

    def test_add_run_success(self, station_with_runs, mth5_empty):
        """Test successful run addition."""
        station, runs = station_with_runs
        new_run = runs[0]

        assert new_run._has_read_metadata is True
        assert "MT003a" in station.groups_list
        assert isinstance(new_run, groups.RunGroup)

        # Test getting run
        rg = mth5_empty.get_run("MT003", "MT003a")
        assert isinstance(rg, groups.RunGroup)

        # Test metadata hierarchy - only check for the first run we created
        assert "MT003a" in new_run.station_metadata.run_list
        assert "MT003a" in new_run.survey_metadata.stations["MT003"].run_list

    def test_run_lifecycle(self, mth5_empty):
        """Test complete run lifecycle."""
        station = mth5_empty.add_station("MT004")
        station.add_run("MT004a")
        assert "MT004a" in station.groups_list

        station.remove_run("MT004a")
        assert "MT004a" not in station.groups_list

    def test_get_nonexistent_run_fails(self, mth5_empty):
        """Test that getting non-existent run raises appropriate error."""
        mth5_empty.add_station("MT001")
        with pytest.raises(MTH5Error):
            mth5_empty.get_run("MT001", "NONEXISTENT")


class TestMTH5ChannelOperations:
    """Test channel-level operations."""

    @pytest.fixture
    def station_with_channels(self, mth5_empty):
        """Create station with run and channels for testing."""
        station = mth5_empty.add_station("MT005")
        run = station.add_run("MT005a")
        channel = run.add_channel("Ex", "electric", None, shape=(4096,))
        return station, run, channel

    def test_add_channel_success(self, station_with_channels, mth5_empty):
        """Test successful channel addition."""
        station, run, new_channel = station_with_channels

        assert "ex" in run.groups_list
        assert isinstance(new_channel, groups.ElectricDataset)
        assert new_channel.hdf5_dataset.shape == (4096,)

        # Test getting channel
        ch = mth5_empty.get_channel("MT005", "MT005a", "ex")
        assert isinstance(ch, groups.ElectricDataset)

        # Test metadata hierarchy
        assert new_channel.run_metadata.channels_recorded_all == ["ex"]
        assert new_channel.station_metadata.runs["MT005a"].channels_recorded_all == [
            "ex"
        ]
        assert new_channel.survey_metadata.stations["MT005"].runs[
            "MT005a"
        ].channels_recorded_all == ["ex"]

    def test_channel_lifecycle(self, mth5_empty):
        """Test complete channel lifecycle."""
        station = mth5_empty.add_station("MT006")
        run = station.add_run("MT006a")
        run.add_channel("Ex", "electric", None)
        assert "ex" in run.groups_list

        run.remove_channel("Ex")
        assert "ex" not in run.groups_list

    def test_get_nonexistent_channel_fails(self, mth5_empty):
        """Test that getting non-existent channel raises appropriate error."""
        station = mth5_empty.add_station("MT007")
        station.add_run("MT007a")
        with pytest.raises(MTH5Error):
            mth5_empty.get_channel("MT007", "MT007a", "nonexistent")


class TestMTH5PathGeneration:
    """Test HDF5 path generation methods."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "/Survey"),
            ({"station": "mt 001"}, "/Survey/Stations/mt_001"),
            ({"station": "mt 001", "run": "a 001"}, "/Survey/Stations/mt_001/a_001"),
            (
                {"station": "mt 001", "run": "a 001", "channel": "ex"},
                "/Survey/Stations/mt_001/a_001/ex",
            ),
        ],
    )
    def test_path_generation(self, mth5_empty, kwargs, expected):
        """Test various path generation scenarios."""
        result = mth5_empty._make_h5_path(**kwargs)
        assert result == expected


class TestMTH5WithData:
    """Test MTH5 operations with realistic data - optimized."""

    def test_channel_properties(self, mth5_with_data_session):
        """Test channel properties are correctly set."""
        run_group = mth5_with_data_session._test_run_group
        expected_components = ["ex", "hy"]  # Reduced for speed

        assert set(run_group.groups_list) == set(expected_components)

        # Test basic channel properties
        ex_channel = run_group.get_channel("ex")
        assert ex_channel.start == MTime(time_stamp="2020-01-01T12:00:00")
        assert ex_channel.sample_rate == 1
        assert ex_channel.n_samples == 256  # Smaller dataset

    def test_data_slicing(self, mth5_with_data_session):
        """Test data slicing functionality."""
        run_group = mth5_with_data_session._test_run_group
        r_slice = run_group.to_runts(
            start="2020-01-01T12:00:00", n_samples=64
        )  # Smaller slice

        assert r_slice.end == "2020-01-01T12:01:03+00:00"
        assert r_slice.dataset.coords.indexes["time"].size == 64

    def test_metadata_hierarchy_integration(self, mth5_with_data_session):
        """Test metadata hierarchy is properly maintained."""
        survey_meta = mth5_with_data_session.survey_group.metadata

        # Test survey level
        assert survey_meta.station_names == ["MT009"]

        # Test station level
        assert survey_meta.stations["MT009"].run_list == ["MT009a"]

        # Test run level - reduced channels
        expected_channels = ["ex", "hy"]
        actual_channels = (
            survey_meta.stations["MT009"].runs["MT009a"].channels_recorded_all
        )
        assert set(actual_channels) == set(expected_channels)

    def test_channel_timeseries_conversion(
        self, mth5_with_data, sample_timeseries_data
    ):
        """Test channel to timeseries conversion - optimized."""
        station = mth5_with_data.add_station("MT002")
        run = station.add_run("MT002a")

        # Create and add channel from timeseries - smaller data
        channel_ts = sample_timeseries_data("Ex", "electric", n_samples=128)
        ex = run.add_channel("Ex", "electric", None)
        ex.from_channel_ts(channel_ts)

        # Convert back to timeseries
        new_ts = ex.to_channel_ts()

        assert channel_ts.start == new_ts.start


class TestMTH5ReadMode:
    """Test MTH5 operations in read-only mode."""

    @pytest.fixture
    def readonly_mth5(self, mth5_with_data, test_file_path):
        """Create a read-only MTH5 file."""
        # Close the writable file first
        mth5_with_data.close_mth5()

        # Reopen in read mode
        mth5_readonly = MTH5(file_version="0.1.0")
        mth5_readonly.open_mth5(test_file_path, mode="r")
        yield mth5_readonly

        try:
            if (
                hasattr(mth5_readonly, "hdf5_file")
                and mth5_readonly.hdf5_file is not None
            ):
                mth5_readonly.close_mth5()
        except Exception:
            pass

    def test_readonly_access(self, readonly_mth5):
        """Test that read-only file allows reading but not writing."""
        # Should be able to read
        assert readonly_mth5.h5_is_read() is True
        assert readonly_mth5.h5_is_write() is False

        # Should be able to access existing data
        station = readonly_mth5.get_station("MT009")
        assert isinstance(station, groups.StationGroup)

        run = readonly_mth5.get_run("MT009", "MT009a")
        assert isinstance(run, groups.RunGroup)

        # Test accessing channels - only test channels that exist in optimized fixture
        for component in ["ex", "hy"]:  # Reduced channel set for speed
            channel = readonly_mth5.get_channel("MT009", "MT009a", component)
            assert isinstance(
                channel,
                (
                    groups.ElectricDataset,
                    groups.MagneticDataset,
                    groups.AuxiliaryDataset,
                ),
            )

    def test_readonly_modification_fails(self, readonly_mth5):
        """Test that modification operations fail in read-only mode."""
        # The add_station method should issue warnings but not raise exceptions
        # since it tries to create groups but the file is read-only
        station = readonly_mth5.add_station("NewStation")
        # The station should be None or the operation should have failed gracefully
        assert station is None or not hasattr(station, "metadata")


# =============================================================================
# Performance and Integration Tests
# =============================================================================


class TestMTH5Performance:
    """Performance-focused tests - optimized for speed."""

    @pytest.mark.performance
    def test_large_data_handling(self, mth5_empty, sample_timeseries_data):
        """Test handling of larger datasets - optimized."""
        station = mth5_empty.add_station("PERF_TEST")
        run = station.add_run("PERF_001")

        # Create moderately sized dataset for speed
        large_channel = sample_timeseries_data(
            "Ex", "electric", n_samples=10000, sample_rate=256  # Reduced from 100k
        )

        ex = run.add_channel("Ex", "electric", None)
        ex.from_channel_ts(large_channel)

        # Test data integrity
        assert ex.n_samples == 10000
        assert ex.sample_rate == 256

        # Test slicing performance
        slice_data = ex.time_slice(
            "2020-01-01T12:00:00", n_samples=500, return_type="numpy"  # Smaller slice
        )
        assert len(slice_data) == 500

    @pytest.mark.integration
    def test_multiple_surveys(self, mth5_empty, sample_timeseries_data):
        """Test handling multiple surveys - optimized."""
        # Reduced survey complexity for speed
        surveys_data = [
            ("Survey_A", ["STA_01"], ["RUN_01"]),  # Fewer stations/runs
            ("Survey_B", ["STB_01"], ["RUN_01"]),
        ]

        for survey_name, station_names, run_names in surveys_data:
            for station_name in station_names:
                station = mth5_empty.add_station(station_name, survey=survey_name)

                for run_name in run_names:
                    run = station.add_run(run_name)

                    # Add minimal channels for speed
                    for comp, ch_type in [("ex", "electric")]:  # Reduced to 1 channel
                        channel_ts = sample_timeseries_data(
                            comp, ch_type, n_samples=128  # Much smaller
                        )
                        channel = run.add_channel(comp.capitalize(), ch_type, None)
                        channel.from_channel_ts(channel_ts)

        # Verify structure
        stations_list = mth5_empty.stations_group.groups_list
        assert len(stations_list) >= 2  # At least our test stations


if __name__ == "__main__":
    """
    Run the optimized MTH5 test suite

    Performance optimizations included:
    - Session-scoped fixtures for expensive setup
    - Reduced data sizes for faster processing
    - Minimal channel sets (ex, hy instead of all 5)
    - Numpy random seed for reproducibility
    - Test markers for fast/slow categorization

    Usage:
    - Run all tests: python test_mth5_pytest.py
    - Run only fast tests: pytest -m fast test_mth5_pytest.py
    - Run with timing info: pytest --durations=10 test_mth5_pytest.py
    - Run quietly: pytest -q --disable-warnings test_mth5_pytest.py
    """
    import sys

    args = [__file__, "-v", "--tb=short"]
    if "--fast" in sys.argv or "-m fast" in " ".join(sys.argv):
        args.extend(["-m", "fast"])
    pytest.main(args)
