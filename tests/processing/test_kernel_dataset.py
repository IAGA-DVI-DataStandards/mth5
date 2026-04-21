# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import pandas as pd
import pytest

from mth5.processing import KERNEL_DATASET_COLUMNS, KernelDataset, RunSummary
from mth5.processing.kernel_dataset import intervals_overlap, overlap


# =============================================================================
# Fixtures using global cache from conftest.py
# =============================================================================


@pytest.fixture
def base_run_summary(global_test12rr_mth5):
    """Create base RunSummary from global MTH5 file."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_test12rr_mth5])
    return run_summary


@pytest.fixture
def run_summary(base_run_summary):
    """Fresh clone of run_summary for each test."""
    return base_run_summary.clone()


@pytest.fixture
def mth5_path(fresh_test12rr_mth5):
    """Create fresh MTH5 file for tests that modify data."""
    # Return the fresh copy from global fixture
    return fresh_test12rr_mth5


@pytest.fixture(scope="session")
def shared_mth5_path(global_test12rr_mth5):
    """Shared MTH5 path for read-only tests (session scope for speed)."""
    from mth5.utils.helpers import close_open_files

    yield global_test12rr_mth5
    close_open_files()


@pytest.fixture(scope="session")
def shared_run_summary(global_test12rr_mth5):
    """Shared RunSummary for read-only tests (session scope for speed)."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_test12rr_mth5])
    return run_summary


@pytest.fixture
def shared_kernel_dataset(shared_run_summary):
    """Create KernelDataset from shared RunSummary for read-only tests."""
    kd = KernelDataset()
    kd.from_run_summary(shared_run_summary.clone(), "test1", "test2")
    return kd


@pytest.fixture
def kernel_dataset(run_summary):
    """Create KernelDataset with test1 local and test2 remote stations."""
    kd = KernelDataset()
    kd.from_run_summary(run_summary, "test1", "test2")
    return kd


@pytest.fixture
def custom_run_summary_data():
    """Create custom DataFrame for specific tests."""
    local = "mt01"
    remote = "mt02"
    return {
        "local": local,
        "remote": remote,
        "df": pd.DataFrame(
            {
                "channel_scale_factors": [1, 1, 1, 1],
                "duration": [0, 0, 0, 0],
                "end": [
                    "2020-01-01T23:59:59",
                    "2020-01-02T02:59:59",
                    "2020-01-01T22:00:00",
                    "2020-01-02T02:00:00.5",
                ],
                "has_data": [True, True, True, True],
                "input_channels": ["hx, hy"] * 4,
                "mth5_path": ["path"] * 2 + ["remote_path"] * 2,
                "n_samples": [86400, 86400, 79200, 7200],
                "output_channels": ["hz, ex, ey"] * 4,
                "run": ["01", "02", "03", "04"],
                "sample_rate": [1, 1, 1, 1],
                "start": [
                    "2020-01-01T00:00:00",
                    "2020-01-02T00:00:00",
                    "2020-01-01T00:00:00",
                    "2020-01-02T00:00:00.5",
                ],
                "station": [local, local, remote, remote],
                "survey": ["test"] * 4,
                "run_hdf5_reference": [None] * 4,
                "station_hdf5_reference": [None] * 4,
            }
        ),
    }


@pytest.fixture
def custom_kernel_dataset(custom_run_summary_data):
    """Create KernelDataset from custom data."""
    rs = RunSummary(df=custom_run_summary_data["df"])
    kd = KernelDataset()
    kd.from_run_summary(
        rs, custom_run_summary_data["local"], custom_run_summary_data["remote"]
    )
    return kd


@pytest.fixture
def fail_run_summary_data():
    """Create DataFrame data that should cause failures."""
    local = "mt01"
    remote = "mt02"
    return {
        "local": local,
        "remote": remote,
        "df": pd.DataFrame(
            {
                "channel_scale_factors": [1, 1, 1, 1],
                "duration": [0, 0, 0, 0],
                "end": [
                    "2020-01-01T23:59:59",
                    "2020-01-01T02:59:59",
                    "2020-02-02T22:00:00",
                    "2020-02-02T02:00:00.5",
                ],
                "has_data": [True, True, True, True],
                "input_channels": ["hx, hy"] * 4,
                "mth5_path": ["path"] * 4,
                "n_samples": [86400, 86400, 79200, 7200],
                "output_channels": ["hz, ex, ey"] * 4,
                "run": ["01", "02", "03", "04"],
                "sample_rate": [1, 4, 1, 4],
                "start": [
                    "2020-01-01T00:00:00",
                    "2020-01-01T00:00:00",
                    "2020-02-02T00:00:00",
                    "2020-02-02T00:00:00.5",
                ],
                "station": [local, local, remote, remote],
                "survey": ["test"] * 4,
                "run_hdf5_reference": [None] * 4,
                "station_hdf5_reference": [None] * 4,
            }
        ),
    }


@pytest.fixture
def overlap_test_data():
    """Create test data for overlap functions."""
    # A day long interval
    ti1_start = pd.Timestamp(1980, 1, 1, 12, 30, 0)
    ti1_end = pd.Timestamp(1980, 1, 2, 12, 30, 0)
    shift_1_hours = 5
    shift_2_hours = 25

    # Shift the interval forward, leave it overlapping
    ti2_start = ti1_start + pd.Timedelta(hours=shift_1_hours)
    ti2_end = ti1_end + pd.Timedelta(hours=shift_1_hours)

    # Shift the interval forward, non-overlapping
    ti3_start = ti1_start + pd.Timedelta(hours=shift_2_hours)
    ti3_end = ti1_end + pd.Timedelta(hours=shift_2_hours)

    return {
        "ti1_start": ti1_start,
        "ti1_end": ti1_end,
        "ti2_start": ti2_start,
        "ti2_end": ti2_end,
        "ti3_start": ti3_start,
        "ti3_end": ti3_end,
        "shift_1_hours": shift_1_hours,
    }


# =============================================================================
# Test Classes and Functions
# =============================================================================


class TestKernelDataset:
    """Test KernelDataset basic functionality."""

    def test_df_operations(self, kernel_dataset):
        """Test basic DataFrame operations."""
        assert kernel_dataset._has_df()
        assert kernel_dataset._df_has_local_station_id
        assert kernel_dataset._df_has_remote_station_id
        assert kernel_dataset.df.shape == (2, 20)

    def test_set_datetime_columns(self, kernel_dataset):
        """Test setting datetime columns with subtests."""
        new_df = kernel_dataset._set_datetime_columns(kernel_dataset.df)

        assert (
            new_df.start.dtype.type == pd.Timestamp
        ), "start column should be Timestamp"
        assert new_df.end.dtype.type == pd.Timestamp, "end column should be Timestamp"

    def test_df_columns(self, kernel_dataset):
        """Test DataFrame has correct columns."""
        assert sorted(KERNEL_DATASET_COLUMNS) == sorted(kernel_dataset.df.columns)

    def test_set_df_failures(self, kernel_dataset):
        """Test DataFrame setter with invalid inputs."""
        with pytest.raises(TypeError):
            kernel_dataset.df = 10

        with pytest.raises(ValueError):
            kernel_dataset.df = pd.DataFrame({"test": [0]})

    def test_exception_from_empty_run_summary(self, run_summary):
        """Test exception when run_summary has no data."""
        rs = run_summary.clone()
        rs.df.loc[:, "has_data"] = False
        rs.drop_no_data_rows()

        kd = KernelDataset()
        with pytest.raises(ValueError):
            kd.from_run_summary(rs, "test1", "test2")

    def test_clone_operations(self, kernel_dataset):
        """Test cloning operations."""
        # Test clone_dataframe
        cloned_df = kernel_dataset.clone_dataframe()
        pd.testing.assert_frame_equal(cloned_df, kernel_dataset.df)

        # Test clone
        clone = kernel_dataset.clone()
        pd.testing.assert_frame_equal(clone.df, kernel_dataset.df)

    def test_mini_summary(self, kernel_dataset):
        """Test mini summary functionality."""
        mini_df = kernel_dataset.mini_summary
        assert sorted(kernel_dataset._mini_summary_columns) == sorted(mini_df.columns)

    def test_station_ids(self, kernel_dataset):
        """Test station ID properties."""
        assert kernel_dataset.local_station_id == "test1"

        # Test setter failures
        with pytest.raises(NameError):
            kernel_dataset.local_station_id = "test3"

        with pytest.raises(NameError):
            kernel_dataset.remote_station_id = "test3"

    def test_string_representation(self, kernel_dataset):
        """Test string representation."""
        mini_df = kernel_dataset.mini_summary
        assert str(mini_df.head()) == str(kernel_dataset)

    def test_channels(self, kernel_dataset):
        """Test channel properties."""
        assert kernel_dataset.input_channels == ["hx", "hy"]
        assert kernel_dataset.output_channels == ["ex", "ey", "hz"]

    def test_dataframe_filtering(self, kernel_dataset):
        """Test local and remote DataFrame filtering."""
        # Test local_df
        local_df = kernel_dataset.local_df
        assert local_df.shape == (1, 20)
        assert len(local_df.station.unique()) == 1
        assert list(local_df.station.unique()) == ["test1"]

        # Test remote_df
        remote_df = kernel_dataset.remote_df
        assert remote_df.shape == (1, 20)
        assert len(remote_df.station.unique()) == 1
        assert list(remote_df.station.unique()) == ["test2"]

    def test_processing_id(self, kernel_dataset):
        """Test processing ID generation."""
        expected_processing_id = "test1_rr_test2_sr1"
        assert kernel_dataset.processing_id == expected_processing_id

    def test_survey_properties(self, kernel_dataset):
        """Test survey-related properties."""
        assert kernel_dataset.local_survey_id == "EMTF Synthetic"
        assert kernel_dataset.sample_rate == 1
        assert kernel_dataset.num_sample_rates == 1

    def test_set_run_times(self, kernel_dataset):
        """Test setting run times."""
        times = {
            "001": {
                "start": "1980-01-01T01:00:00+00:00",
                "end": "1980-01-01T08:00:00+00:00",
            }
        }

        kernel_dataset.set_run_times(times)
        assert (
            kernel_dataset.df.iloc[0].duration == 25200
        ), "Local duration should be 25200"
        assert (
            kernel_dataset.df.iloc[1].duration == 25200
        ), "Remote duration should be 25200"

    def test_set_run_times_failures(self, kernel_dataset):
        """Test set_run_times with invalid inputs."""
        with pytest.raises(TypeError):
            kernel_dataset.set_run_times(10)

        with pytest.raises(TypeError):
            kernel_dataset.set_run_times({"001": 10})

        with pytest.raises(KeyError):
            kernel_dataset.set_run_times({"001": {"a": 1}})


class TestOverlapFunctions:
    """Test overlap utility functions."""

    def test_overlaps_boolean(self, overlap_test_data):
        """Test boolean overlap detection."""
        data = overlap_test_data

        # Test overlapping intervals
        assert intervals_overlap(
            data["ti1_start"], data["ti1_end"], data["ti2_start"], data["ti2_end"]
        )

        # Test non-overlapping intervals
        assert not intervals_overlap(
            data["ti1_start"], data["ti1_end"], data["ti3_start"], data["ti3_end"]
        )

    def test_overlap_returns_interval(self, overlap_test_data):
        """Test overlap function returns correct intervals."""
        data = overlap_test_data

        # This test corresponds to the second line in the if/elif logic
        result = overlap(
            data["ti1_start"], data["ti1_end"], data["ti2_start"], data["ti2_end"]
        )

        expected_start = data["ti1_start"] + pd.Timedelta(hours=data["shift_1_hours"])
        assert result[0] == expected_start
        assert result[1] == data["ti1_end"]


class TestKernelDatasetMethods:
    """Test KernelDataset methods with custom data."""

    def test_from_run_summary_properties(
        self, custom_kernel_dataset, custom_run_summary_data
    ):
        """Test properties set correctly from run_summary."""
        kd = custom_kernel_dataset
        data = custom_run_summary_data

        assert kd.local_station_id == data["local"]
        assert kd.remote_station_id == data["remote"]
        assert not (kd.df.duration == 0).all()  # Should have duration calculated
        assert sorted(kd.df.columns) == sorted(KERNEL_DATASET_COLUMNS)

    def test_mth5_paths(self, custom_kernel_dataset):
        """Test MTH5 path properties."""
        kd = custom_kernel_dataset

        assert kd.local_mth5_path == Path("path")
        assert kd.remote_mth5_path == Path("remote_path")
        assert not kd.has_local_mth5()  # Paths don't exist
        assert not kd.has_remote_mth5()  # Paths don't exist

    def test_sample_rate_properties(self, custom_kernel_dataset):
        """Test sample rate properties."""
        kd = custom_kernel_dataset

        assert kd.num_sample_rates == 1
        assert kd.sample_rate == 1

    def test_data_manipulation(self, custom_kernel_dataset):
        """Test data manipulation methods."""
        kd = custom_kernel_dataset

        # Test drop_runs_shorter_than
        kd.drop_runs_shorter_than(8000)
        assert kd.df.shape == (2, 20)

        # Test survey_id
        assert kd.local_survey_id == "test"

    def test_update_duration_column(self, custom_kernel_dataset):
        """Test duration column update."""
        kd = custom_kernel_dataset

        new_df = kd._update_duration_column(inplace=False)
        assert (new_df.duration == kd.df.duration).all()

    def test_station_id_setter_failures(self, custom_kernel_dataset):
        """Test station ID setter failures."""
        kd = custom_kernel_dataset

        with pytest.raises(NameError):
            kd.local_station_id = "mt03"

        with pytest.raises(NameError):
            kd.remote_station_id = "mt03"


class TestKernelDatasetMethodsFail:
    """Test KernelDataset methods that should fail."""

    def test_from_run_summary_fail(self, fail_run_summary_data):
        """Test from_run_summary with incompatible data."""
        data = fail_run_summary_data
        rs = RunSummary(df=data["df"])
        kd = KernelDataset()

        with pytest.raises(ValueError):
            kd.from_run_summary(rs, data["local"], data["remote"])

    def test_multiple_sample_rates(self, fail_run_summary_data):
        """Test behavior with multiple sample rates."""
        data = fail_run_summary_data
        rs = RunSummary(df=data["df"])
        kd = KernelDataset()
        kd.from_run_summary(rs, data["local"])

        assert kd.num_sample_rates == 2

    def test_sample_rate_fail(self, fail_run_summary_data):
        """Test sample_rate property with multiple rates."""
        data = fail_run_summary_data
        rs = RunSummary(df=data["df"])
        kd = KernelDataset()
        kd.from_run_summary(rs, data["local"])

        with pytest.raises(NotImplementedError):
            _ = kd.sample_rate


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize("bad_value", [10, "string", None, [1, 2, 3]])
def test_df_setter_type_validation(kernel_dataset, bad_value):
    """Test DataFrame setter with various invalid types."""
    if bad_value is None:
        # None is actually allowed
        kernel_dataset.df = bad_value
        assert kernel_dataset.df is None
    else:
        with pytest.raises(TypeError):
            kernel_dataset.df = bad_value


@pytest.mark.parametrize(
    "invalid_df",
    [
        pd.DataFrame({"wrong": [1, 2]}),
        pd.DataFrame({"also_wrong": ["a", "b", "c"]}),
        pd.DataFrame(),  # Empty DataFrame
    ],
)
def test_df_setter_invalid_dataframes(kernel_dataset, invalid_df):
    """Test DataFrame setter with invalid DataFrames."""
    with pytest.raises(ValueError):
        kernel_dataset.df = invalid_df


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_workflow(mth5_path):
    """Test complete workflow from MTH5 to KernelDataset."""
    # Create run summary
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    # Create kernel dataset
    kd = KernelDataset()
    kd.from_run_summary(run_summary, "test1", "test2")

    # Verify complete setup
    assert kd._has_df()
    assert kd.local_station_id == "test1"
    assert kd.remote_station_id == "test2"
    assert len(kd.input_channels) > 0
    assert len(kd.output_channels) > 0
    assert kd.df is not None
    assert kd.df.shape[0] > 0  # Has data

    # Test operations work
    mini = kd.mini_summary
    assert kd.df is not None
    assert mini.shape[0] == kd.df.shape[0]

    local_df = kd.local_df
    remote_df = kd.remote_df
    assert kd.df is not None
    assert local_df is not None
    assert remote_df is not None
    assert local_df.shape[0] + remote_df.shape[0] == kd.df.shape[0]


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.performance
def test_large_dataset_performance(custom_run_summary_data):
    """Test performance with larger datasets."""
    # Create larger dataset but keep the original station structure for valid overlap
    data = custom_run_summary_data
    large_df = pd.concat([data["df"]] * 25, ignore_index=True)  # Smaller multiplier

    # Keep the original local/remote structure to ensure overlap
    # Just modify the run IDs to make them unique
    for i in range(len(large_df)):
        large_df.loc[i, "run"] = f"{i:03d}"

    rs = RunSummary(df=large_df)
    kd = KernelDataset()

    # This should complete quickly - use original local/remote stations
    import time

    start = time.time()
    kd.from_run_summary(rs, data["local"], data["remote"])
    end = time.time()

    assert end - start < 10.0  # Should complete in under 10 seconds
    assert kd._has_df()


# =============================================================================
# Integration Test for initialize_dataframe_for_processing
# =============================================================================


def test_initialize_dataframe_for_processing(mth5_path):
    """Test KernelDataset.initialize_dataframe_for_processing end-to-end."""
    # Create run summary from the provided MTH5 file
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    # Create KernelDataset and initialize from run summary
    kd = KernelDataset()
    # Use the first two stations as local and remote (if available)
    stations = run_summary.df.station.unique()
    if len(stations) < 2:
        # Single station test
        kd.from_run_summary(run_summary, stations[0])
    else:
        kd.from_run_summary(run_summary, stations[0], stations[1])

    # Open MTH5 files for the dataset
    kd.initialize_mth5s()

    # Run the function under test
    kd.initialize_dataframe_for_processing()

    # Check that run_hdf5_reference and run_dataarray columns are populated (not all None)
    assert "run_hdf5_reference" in kd.df.columns
    assert "run_dataarray" in kd.df.columns
    assert (
        kd.df["run_hdf5_reference"].notnull().any()
    ), "run_hdf5_reference should not be all None"
    assert (
        kd.df["run_dataarray"].notnull().any()
    ), "run_dataarray should not be all None"


if __name__ == "__main__":
    pytest.main([__file__])
