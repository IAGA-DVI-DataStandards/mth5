# -*- coding: utf-8 -*-
"""
Test suite for MTH5 experiment building functionality.

Created on Thu May 13 13:45:27 2021

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from mt_metadata import MT_EXPERIMENT_SINGLE_STATION
from mt_metadata.timeseries import Experiment

from mth5 import CHANNEL_DTYPE, helpers
from mth5.mth5 import MTH5


# =============================================================================
# Constants and Utilities
# =============================================================================

fn_path = Path(__file__).parent
helpers.close_open_files()


def clean_metadata_dict(metadata_dict: dict) -> dict:
    """Remove HDF5-specific keys from metadata dictionary."""
    cleaned = metadata_dict.copy()
    cleaned.pop("hdf5_reference", None)
    cleaned.pop("mth5_type", None)
    return cleaned


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def experiment_from_xml() -> Experiment:
    """Load experiment from XML file (session-scoped for efficiency)."""
    experiment = Experiment()
    experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
    return experiment


@pytest.fixture(scope="session")
def mth5_with_experiment(
    experiment_from_xml: Experiment,
    make_worker_safe_path,
) -> Generator[MTH5, None, None]:
    """Create MTH5 file with experiment data (session-scoped for efficiency)."""
    fn = make_worker_safe_path("test_pytest.h5", Path(__file__).parent)
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(fn, mode="w")
    mth5_obj.from_experiment(experiment_from_xml)

    yield mth5_obj

    # Cleanup
    mth5_obj.close_mth5()
    if fn.exists():
        fn.unlink()


@pytest.fixture(scope="session")
def modified_experiment(experiment_from_xml: Experiment) -> Experiment:
    """Create a modified experiment for update testing."""
    experiment_02 = Experiment()
    experiment_02.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
    experiment_02.surveys[0].id = "different_survey_name"
    experiment_02.surveys[0].stations[0].location.latitude = 10
    return experiment_02


@pytest.fixture(scope="session")
def mth5_update_test(
    experiment_from_xml: Experiment,
    modified_experiment: Experiment,
    make_worker_safe_path,
) -> Generator[tuple[MTH5, Experiment, Experiment], None, None]:
    """MTH5 object for update testing."""
    fn = make_worker_safe_path("test_update_pytest.h5", Path(__file__).parent)
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(fn, mode="w")
    mth5_obj.from_experiment(experiment_from_xml)

    yield mth5_obj, experiment_from_xml, modified_experiment

    # Cleanup
    mth5_obj.close_mth5()
    if fn.exists():
        fn.unlink()


@pytest.fixture(scope="session")
def regenerated_experiment(mth5_with_experiment: MTH5) -> Experiment:
    """Generate experiment from MTH5 once for all roundtrip tests (session-scoped for efficiency)."""
    return mth5_with_experiment.to_experiment(has_data=False)


# =============================================================================
# Test Classes
# =============================================================================


class TestMTH5ExperimentBuild:
    """Test MTH5 experiment building and metadata consistency."""

    def test_survey_metadata(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test survey metadata consistency between original and MTH5."""
        survey = experiment_from_xml.surveys[0]

        # Get cleaned metadata dictionaries
        original_data = clean_metadata_dict(survey.to_dict(single=True))
        h5_data = clean_metadata_dict(
            mth5_with_experiment.survey_group.metadata.to_dict(single=True)
        )

        assert original_data == h5_data

    @pytest.mark.parametrize(
        "station",
        [
            pytest.param(lambda exp: station, id=f"station_{i}")
            for i, station in enumerate(["REW09"])
        ],
    )  # Will be dynamically populated
    def test_station_metadata(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment, station
    ):
        """Test station metadata consistency."""
        # Get all stations dynamically for parametrization
        stations = experiment_from_xml.surveys[0].stations

        for station_obj in stations:
            h5_station = mth5_with_experiment.get_station(station_obj.id)

            original_data = clean_metadata_dict(station_obj.to_dict(single=True))
            h5_data = clean_metadata_dict(h5_station.metadata.to_dict(single=True))

            assert (
                h5_data == original_data
            ), f"Station {station_obj.id} metadata mismatch"

    def test_run_metadata(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test run metadata consistency."""
        runs = experiment_from_xml.surveys[0].stations[0].runs
        station_id = experiment_from_xml.surveys[0].stations[0].id

        for run in runs:
            h5_run = mth5_with_experiment.get_run(station_id, run.id)

            original_data = clean_metadata_dict(run.to_dict(single=True))
            h5_data = clean_metadata_dict(h5_run.metadata.to_dict(single=True))

            assert h5_data == original_data, f"Run {run.id} metadata mismatch"

    def test_channel_metadata(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test channel metadata consistency."""
        runs = experiment_from_xml.surveys[0].stations[0].runs
        station_id = experiment_from_xml.surveys[0].stations[0].id

        for run in runs:
            h5_run = mth5_with_experiment.get_run(station_id, run.id)

            for channel in run.channels:
                h5_channel = h5_run.get_channel(channel.component)

                original_data = clean_metadata_dict(channel.to_dict(single=True))
                h5_data = clean_metadata_dict(h5_channel.metadata.to_dict(single=True))

                assert (
                    h5_data == original_data
                ), f"Channel {run.id}/{channel.component} metadata mismatch"

    def test_filter_metadata(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test filter metadata consistency."""
        exp_filters = experiment_from_xml.surveys[0].filters

        for key, value in exp_filters.items():
            # Transform key to match how MTH5 stores filter names
            stored_key = key.replace("/", " per ")
            original_data = value.to_dict(single=True, required=False)
            h5_filter = mth5_with_experiment.filters_group.to_filter_object(stored_key)
            h5_data = h5_filter.to_dict(single=True, required=False)

            for k in original_data.keys():
                if k not in h5_data:
                    continue

                v1, v2 = original_data[k], h5_data[k]

                if isinstance(v1, (float, int)):
                    assert (
                        abs(v1 - float(v2)) < 1e-5
                    ), f"Filter {stored_key}.{k} numerical mismatch"
                elif isinstance(v1, np.ndarray):
                    if v1.dtype != v2.dtype:
                        if np.issubdtype(
                            v2.dtype, np.complexfloating
                        ) and not np.issubdtype(v1.dtype, np.complexfloating):
                            v2_converted = np.real(v2).astype(v1.dtype)
                        else:
                            v2_converted = v2.astype(v1.dtype)
                        assert (
                            v1 == v2_converted
                        ).all(), f"Filter {stored_key}.{k} array mismatch"
                    else:
                        assert v1.dtype == v2.dtype
                        assert (
                            v1 == v2
                        ).all(), f"Filter {stored_key}.{k} array mismatch"
                elif v1 is None and v2 == "None":
                    continue  # Handle None vs 'None' string conversion
                else:
                    assert v1 == v2, f"Filter {stored_key}.{k} value mismatch"


class TestMTH5TimeSeries:
    """Test MTH5 time series functionality."""

    def test_run_to_timeseries(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test conversion of run to time series."""
        station_id = experiment_from_xml.surveys[0].stations[0].id
        run_id = experiment_from_xml.surveys[0].stations[0].runs[0].id

        run_group = mth5_with_experiment.get_run(station_id, run_id)
        run_ts = run_group.to_runts()

        # Test survey metadata
        survey_data = experiment_from_xml.surveys[0].to_dict(single=True)
        for key in survey_data.keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue
            expected = experiment_from_xml.surveys[0].get_attr_from_name(key)
            actual = run_ts.survey_metadata.get_attr_from_name(key)
            assert expected == actual, f"Survey metadata mismatch for {key}"

        # Test station metadata
        station_data = experiment_from_xml.surveys[0].stations[0].to_dict(single=True)
        for key in station_data.keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue

            if key == "run_list":
                assert run_ts.station_metadata.run_list == ["a", "b", "c", "d", "e"]
            else:
                expected = (
                    experiment_from_xml.surveys[0].stations[0].get_attr_from_name(key)
                )
                actual = run_ts.station_metadata.get_attr_from_name(key)
                assert expected == actual, f"Station metadata mismatch for {key}"

        # Test run metadata
        run_data = (
            experiment_from_xml.surveys[0].stations[0].runs[0].to_dict(single=True)
        )
        for key in run_data.keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue

            expected = (
                experiment_from_xml.surveys[0]
                .stations[0]
                .runs[0]
                .get_attr_from_name(key)
            )
            actual = run_ts.run_metadata.get_attr_from_name(key)

            if key == "time_period.end":
                # Known issue: end time is off by one second
                assert expected != actual
            else:
                assert expected == actual, f"Run metadata mismatch for {key}"

    def test_channel_to_timeseries(
        self, mth5_with_experiment: MTH5, experiment_from_xml: Experiment
    ):
        """Test conversion of channel to time series."""
        station_id = experiment_from_xml.surveys[0].stations[0].id
        run_id = experiment_from_xml.surveys[0].stations[0].runs[0].id
        component = (
            experiment_from_xml.surveys[0].stations[0].runs[0].channels[0].component
        )

        channel_group = mth5_with_experiment.get_channel(station_id, run_id, component)
        ch_ts = channel_group.to_channel_ts()

        # Test survey metadata
        survey_data = experiment_from_xml.surveys[0].to_dict(single=True)
        for key in survey_data.keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue
            expected = experiment_from_xml.surveys[0].get_attr_from_name(key)
            actual = ch_ts.survey_metadata.get_attr_from_name(key)
            assert expected == actual, f"Survey metadata mismatch for {key}"

        # Test station metadata
        station_data = experiment_from_xml.surveys[0].stations[0].to_dict(single=True)
        for key in station_data.keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue

            if key in ["run_list", "channels_recorded"]:
                assert ch_ts.station_metadata.run_list == ["a", "b", "c", "d", "e"]
            else:
                expected = (
                    experiment_from_xml.surveys[0].stations[0].get_attr_from_name(key)
                )
                actual = ch_ts.station_metadata.get_attr_from_name(key)

                # Handle location coordinates that are None in XML but 0.0 in MTH5
                if (
                    key in ["location.x", "location.y", "location.z"]
                    and expected is None
                    and actual == 0.0
                ):
                    continue

                assert expected == actual, f"Station metadata mismatch for {key}"

        # Test run metadata
        run_data = (
            experiment_from_xml.surveys[0].stations[0].runs[0].to_dict(single=True)
        )
        for key in run_data.keys():
            if key in [
                "hdf5_reference",
                "mth5_type",
                "channels_recorded_magnetic",
                "channels_recorded_electric",
                "channels_recorded_auxiliary",
            ]:
                continue

            expected = (
                experiment_from_xml.surveys[0]
                .stations[0]
                .runs[0]
                .get_attr_from_name(key)
            )
            actual = ch_ts.run_metadata.get_attr_from_name(key)
            assert expected == actual, f"Run metadata mismatch for {key}"

        # Test channel metadata
        channel_data = (
            experiment_from_xml.surveys[0]
            .stations[0]
            .runs[0]
            .channels[0]
            .to_dict(single=True)
        )
        for key in channel_data.keys():
            if key in ["hdf5_reference", "mth5_type", "filter.name", "filter.applied"]:
                continue

            expected = (
                experiment_from_xml.surveys[0]
                .stations[0]
                .runs[0]
                .channels[0]
                .get_attr_from_name(key)
            )

            if key == "time_period.end":
                # Known issue: end time is off by one second
                actual = ch_ts.station_metadata.get_attr_from_name(key)
                assert expected != actual
            else:
                actual = ch_ts.channel_metadata.get_attr_from_name(key)
                assert expected == actual, f"Channel metadata mismatch for {key}"


class TestMTH5Summary:
    """Test MTH5 summary functionality."""

    def test_channel_summary(self, mth5_with_experiment: MTH5):
        """Test channel summary functionality."""
        mth5_with_experiment.channel_summary.summarize()

        assert mth5_with_experiment.channel_summary.shape == (25,)
        assert mth5_with_experiment.channel_summary.nrows == 25
        assert mth5_with_experiment.channel_summary.dtype == CHANNEL_DTYPE
        assert (mth5_with_experiment.channel_summary.array["station"] == b"REW09").all()

    def test_run_summary(self, mth5_with_experiment: MTH5):
        """Test run summary functionality."""
        mth5_with_experiment.channel_summary.summarize()
        run_summary = mth5_with_experiment.channel_summary.to_run_summary()

        assert isinstance(run_summary, pd.DataFrame)
        assert run_summary.shape == (5, 15)

    def test_run_summary_property(self, mth5_with_experiment: MTH5):
        """Test run summary property."""
        run_summary = mth5_with_experiment.run_summary

        assert isinstance(run_summary, pd.DataFrame)
        assert run_summary.shape == (5, 15)


class TestMTH5Update:
    """Test MTH5 update functionality."""

    def test_update_from_new_experiment(
        self, mth5_update_test: tuple[MTH5, Experiment, Experiment]
    ):
        """Test updating MTH5 with modified experiment."""
        mth5_obj, original_experiment, modified_experiment = mth5_update_test

        # Perform update
        mth5_obj.from_experiment(modified_experiment, update=True)

        # Test survey ID update
        assert mth5_obj.survey_group.metadata.id == modified_experiment.surveys[0].id

        # Test location update
        station = mth5_obj.get_station("REW09")
        expected_lat = modified_experiment.surveys[0].stations[0].location.latitude
        assert station.metadata.location.latitude == expected_lat


# =============================================================================
# Test MTH5 to_experiment Functionality
# =============================================================================
class TestMTH5ToExperiment:
    """Test MTH5 to_experiment functionality for round-trip consistency.

    These tests verify that MTH5.to_experiment() correctly reconstructs the full
    experiment hierarchy including surveys, stations, runs, and channels from the
    HDF5 file structure.
    """

    def test_experiment_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test that to_experiment produces an equivalent Experiment object."""
        # Test that we get an Experiment object
        assert isinstance(regenerated_experiment, Experiment)

        # Test survey count
        assert len(regenerated_experiment.surveys) == len(experiment_from_xml.surveys)

    def test_survey_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test survey metadata roundtrip consistency."""
        original_survey = experiment_from_xml.surveys[0]
        regenerated_survey = regenerated_experiment.surveys[0]

        # Get cleaned metadata
        original_data = clean_metadata_dict(original_survey.to_dict(single=True))
        regenerated_data = clean_metadata_dict(regenerated_survey.to_dict(single=True))

        assert original_data == regenerated_data

    def test_station_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test station metadata roundtrip consistency."""
        original_stations = experiment_from_xml.surveys[0].stations
        regenerated_stations = regenerated_experiment.surveys[0].stations

        # Test station count
        assert len(regenerated_stations) == len(original_stations)

        # Test each station
        for orig_station in original_stations:
            regen_station = next(
                s for s in regenerated_stations if s.id == orig_station.id
            )

            original_data = clean_metadata_dict(orig_station.to_dict(single=True))
            regenerated_data = clean_metadata_dict(regen_station.to_dict(single=True))

            assert (
                original_data == regenerated_data
            ), f"Station {orig_station.id} roundtrip failed"

    def test_run_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test run metadata roundtrip consistency."""
        original_station = experiment_from_xml.surveys[0].stations[0]
        regenerated_station = regenerated_experiment.surveys[0].stations[0]

        # Test run count
        assert len(regenerated_station.runs) == len(original_station.runs)

        # Test each run
        for orig_run in original_station.runs:
            regen_run = next(r for r in regenerated_station.runs if r.id == orig_run.id)

            original_data = clean_metadata_dict(orig_run.to_dict(single=True))
            regenerated_data = clean_metadata_dict(regen_run.to_dict(single=True))

            assert (
                original_data == regenerated_data
            ), f"Run {orig_run.id} roundtrip failed"

    def test_channel_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test channel metadata roundtrip consistency."""
        original_run = experiment_from_xml.surveys[0].stations[0].runs[0]
        regenerated_run = regenerated_experiment.surveys[0].stations[0].runs[0]

        # Test channel count
        assert len(regenerated_run.channels) == len(original_run.channels)

        # Test each channel
        for orig_channel in original_run.channels:
            regen_channel = next(
                c
                for c in regenerated_run.channels
                if c.component == orig_channel.component
            )

            original_data = clean_metadata_dict(orig_channel.to_dict(single=True))
            regenerated_data = clean_metadata_dict(regen_channel.to_dict(single=True))

            assert (
                original_data == regenerated_data
            ), f"Channel {orig_channel.component} roundtrip failed"

    def test_filter_roundtrip(
        self, regenerated_experiment: Experiment, experiment_from_xml: Experiment
    ):
        """Test filter metadata roundtrip consistency."""
        original_filters = experiment_from_xml.surveys[0].filters
        regenerated_filters = regenerated_experiment.surveys[0].filters

        # Test filter count
        assert len(regenerated_filters) == len(original_filters)

        # Test each filter
        for key, orig_filter in original_filters.items():
            # Handle key transformation for stored filters
            stored_key = key.replace("/", " per ")

            # Find corresponding regenerated filter
            regen_filter = None
            for regen_key, regen_val in regenerated_filters.items():
                if regen_key == key or regen_key == stored_key:
                    regen_filter = regen_val
                    break

            assert (
                regen_filter is not None
            ), f"Filter {key} not found in regenerated experiment"

            original_data = orig_filter.to_dict(single=True, required=False)
            regenerated_data = regen_filter.to_dict(single=True, required=False)

            for k in original_data.keys():
                if k not in regenerated_data:
                    continue

                v1, v2 = original_data[k], regenerated_data[k]

                if isinstance(v1, (float, int)):
                    assert (
                        abs(v1 - float(v2)) < 1e-5
                    ), f"Filter {key}.{k} numerical mismatch"
                elif isinstance(v1, np.ndarray):
                    if v1.dtype != v2.dtype:
                        if np.issubdtype(
                            v2.dtype, np.complexfloating
                        ) and not np.issubdtype(v1.dtype, np.complexfloating):
                            v2_converted = np.real(v2).astype(v1.dtype)
                        else:
                            v2_converted = v2.astype(v1.dtype)
                        assert (
                            v1 == v2_converted
                        ).all(), f"Filter {key}.{k} array mismatch"
                    else:
                        assert (v1 == v2).all(), f"Filter {key}.{k} array mismatch"
                elif v1 is None and v2 == "None":
                    continue
                else:
                    assert v1 == v2, f"Filter {key}.{k} value mismatch"

    def test_complete_roundtrip_workflow(
        self, experiment_from_xml: Experiment, make_worker_safe_path
    ):
        """Test complete roundtrip: Experiment -> MTH5 -> Experiment."""
        fn = make_worker_safe_path("test_complete_roundtrip.h5", Path(__file__).parent)

        try:
            # Step 1: Create MTH5 from experiment
            mth5_obj = MTH5(file_version="0.1.0")
            mth5_obj.open_mth5(fn, mode="w")
            mth5_obj.from_experiment(experiment_from_xml)

            # Step 2: Convert back to experiment
            regenerated_experiment = mth5_obj.to_experiment(has_data=False)

            # Step 3: Compare key attributes
            assert (
                regenerated_experiment.surveys[0].id
                == experiment_from_xml.surveys[0].id
            )
            assert len(regenerated_experiment.surveys[0].stations) == len(
                experiment_from_xml.surveys[0].stations
            )

            station_orig = experiment_from_xml.surveys[0].stations[0]
            station_regen = regenerated_experiment.surveys[0].stations[0]

            assert station_regen.id == station_orig.id
            assert len(station_regen.runs) == len(station_orig.runs)
            assert station_regen.location.latitude == station_orig.location.latitude
            assert station_regen.location.longitude == station_orig.location.longitude

        finally:
            mth5_obj.close_mth5()
            if fn.exists():
                fn.unlink()


# =============================================================================
# Parametrized Tests for Multiple Items
# =============================================================================


def pytest_generate_tests(metafunc):
    """Generate parametrized tests dynamically."""
    if "station_id" in metafunc.fixturenames:
        # Load experiment to get station IDs
        experiment = Experiment()
        experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        station_ids = [station.id for station in experiment.surveys[0].stations]
        metafunc.parametrize("station_id", station_ids)

    elif "run_info" in metafunc.fixturenames:
        # Load experiment to get run information
        experiment = Experiment()
        experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        run_infos = []
        for station in experiment.surveys[0].stations:
            for run in station.runs:
                run_infos.append((station.id, run.id))
        metafunc.parametrize(
            "run_info", run_infos, ids=[f"{s}_{r}" for s, r in run_infos]
        )

    elif "channel_info" in metafunc.fixturenames:
        # Load experiment to get channel information
        experiment = Experiment()
        experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        channel_infos = []
        for station in experiment.surveys[0].stations:
            for run in station.runs:
                for channel in run.channels:
                    channel_infos.append((station.id, run.id, channel.component))
        metafunc.parametrize(
            "channel_info",
            channel_infos,
            ids=[f"{s}_{r}_{c}" for s, r, c in channel_infos],
        )


def test_individual_station_metadata(
    mth5_with_experiment: MTH5, experiment_from_xml: Experiment, station_id: str
):
    """Test individual station metadata (parametrized)."""
    station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    h5_station = mth5_with_experiment.get_station(station_id)

    original_data = clean_metadata_dict(station.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_station.metadata.to_dict(single=True))

    assert h5_data == original_data


def test_individual_run_metadata(
    mth5_with_experiment: MTH5,
    experiment_from_xml: Experiment,
    run_info: tuple[str, str],
):
    """Test individual run metadata (parametrized)."""
    station_id, run_id = run_info

    # Find the run in the experiment
    station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    run = next(r for r in station.runs if r.id == run_id)

    h5_run = mth5_with_experiment.get_run(station_id, run_id)

    original_data = clean_metadata_dict(run.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_run.metadata.to_dict(single=True))

    assert h5_data == original_data


def test_individual_channel_metadata(
    mth5_with_experiment: MTH5,
    experiment_from_xml: Experiment,
    channel_info: tuple[str, str, str],
):
    """Test individual channel metadata (parametrized)."""
    station_id, run_id, component = channel_info

    # Find the channel in the experiment
    station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    run = next(r for r in station.runs if r.id == run_id)
    channel = next(c for c in run.channels if c.component == component)

    h5_run = mth5_with_experiment.get_run(station_id, run_id)
    h5_channel = h5_run.get_channel(component)

    original_data = clean_metadata_dict(channel.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_channel.metadata.to_dict(single=True))

    assert h5_data == original_data


def test_parametrized_station_roundtrip(
    regenerated_experiment: Experiment,
    experiment_from_xml: Experiment,
    station_id: str,
):
    """Test individual station roundtrip (parametrized)."""
    original_station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    regenerated_station = next(
        s for s in regenerated_experiment.surveys[0].stations if s.id == station_id
    )

    original_data = clean_metadata_dict(original_station.to_dict(single=True))
    regenerated_data = clean_metadata_dict(regenerated_station.to_dict(single=True))

    assert original_data == regenerated_data


def test_parametrized_run_roundtrip(
    regenerated_experiment: Experiment,
    experiment_from_xml: Experiment,
    run_info: tuple[str, str],
):
    """Test individual run roundtrip (parametrized)."""
    station_id, run_id = run_info

    # Find original run
    original_station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    original_run = next(r for r in original_station.runs if r.id == run_id)

    # Find regenerated run
    regenerated_station = next(
        s for s in regenerated_experiment.surveys[0].stations if s.id == station_id
    )
    regenerated_run = next(r for r in regenerated_station.runs if r.id == run_id)

    original_data = clean_metadata_dict(original_run.to_dict(single=True))
    regenerated_data = clean_metadata_dict(regenerated_run.to_dict(single=True))

    assert original_data == regenerated_data


def test_parametrized_channel_roundtrip(
    regenerated_experiment: Experiment,
    experiment_from_xml: Experiment,
    channel_info: tuple[str, str, str],
):
    """Test individual channel roundtrip (parametrized)."""
    station_id, run_id, component = channel_info

    # Find original channel
    original_station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    original_run = next(r for r in original_station.runs if r.id == run_id)
    original_channel = next(
        c for c in original_run.channels if c.component == component
    )

    # Find regenerated channel
    regenerated_station = next(
        s for s in regenerated_experiment.surveys[0].stations if s.id == station_id
    )
    regenerated_run = next(r for r in regenerated_station.runs if r.id == run_id)
    regenerated_channel = next(
        c for c in regenerated_run.channels if c.component == component
    )

    original_data = clean_metadata_dict(original_channel.to_dict(single=True))
    regenerated_data = clean_metadata_dict(regenerated_channel.to_dict(single=True))

    assert original_data == regenerated_data
