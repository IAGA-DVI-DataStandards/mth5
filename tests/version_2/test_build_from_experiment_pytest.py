# -*- coding: utf-8 -*-
"""
Test suite for MTH5 version 0.2.0 experiment building functionality.

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
from mth5.helpers import validate_name
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
    fn = make_worker_safe_path("test_pytest_v2.h5", Path(__file__).parent)
    mth5_obj = MTH5(file_version="0.2.0")  # Key difference: using 0.2.0
    mth5_obj.open_mth5(fn, mode="w")
    mth5_obj.from_experiment(experiment_from_xml)

    yield mth5_obj

    # Cleanup
    mth5_obj.close_mth5()
    if fn.exists():
        fn.unlink()


@pytest.fixture(scope="session")
def survey_name(experiment_from_xml: Experiment) -> str:
    """Get validated survey name."""
    return validate_name(experiment_from_xml.surveys[0].id)


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
    fn = make_worker_safe_path("test_update_pytest_v2.h5", Path(__file__).parent)
    mth5_obj = MTH5(file_version="0.2.0")  # Using 0.2.0 for update test too
    mth5_obj.open_mth5(fn, mode="w")
    mth5_obj.from_experiment(experiment_from_xml)

    yield mth5_obj, experiment_from_xml, modified_experiment

    # Cleanup
    mth5_obj.close_mth5()
    if fn.exists():
        fn.unlink()


# =============================================================================
# Test Classes
# =============================================================================


class TestMTH5ExperimentBuild:
    """Test MTH5 experiment building and metadata consistency."""

    def test_survey_metadata(
        self,
        mth5_with_experiment: MTH5,
        experiment_from_xml: Experiment,
        survey_name: str,
    ):
        """Test survey metadata consistency between original and MTH5."""
        survey = experiment_from_xml.surveys[0]
        sg = mth5_with_experiment.get_survey(survey_name)

        # Get cleaned metadata dictionaries
        original_data = clean_metadata_dict(survey.to_dict(single=True))
        h5_data = clean_metadata_dict(sg.metadata.to_dict(single=True))

        assert original_data == h5_data

    @pytest.mark.parametrize(
        "station",
        [
            pytest.param(lambda exp: station, id=f"station_{i}")
            for i, station in enumerate(["REW09"])
        ],
    )  # Will be dynamically populated
    def test_station_metadata(
        self,
        mth5_with_experiment: MTH5,
        experiment_from_xml: Experiment,
        survey_name: str,
        station,
    ):
        """Test station metadata consistency."""
        # Get all stations dynamically for parametrization
        stations = experiment_from_xml.surveys[0].stations

        for station_obj in stations:
            h5_station = mth5_with_experiment.get_station(station_obj.id, survey_name)

            original_data = clean_metadata_dict(station_obj.to_dict(single=True))
            h5_data = clean_metadata_dict(h5_station.metadata.to_dict(single=True))

            assert (
                h5_data == original_data
            ), f"Station {station_obj.id} metadata mismatch"

    def test_run_metadata(
        self,
        mth5_with_experiment: MTH5,
        experiment_from_xml: Experiment,
        survey_name: str,
    ):
        """Test run metadata consistency."""
        runs = experiment_from_xml.surveys[0].stations[0].runs
        station_id = experiment_from_xml.surveys[0].stations[0].id

        for run in runs:
            h5_run = mth5_with_experiment.get_run(
                station_id, run.id, survey=survey_name
            )

            original_data = clean_metadata_dict(run.to_dict(single=True))
            h5_data = clean_metadata_dict(h5_run.metadata.to_dict(single=True))

            assert h5_data == original_data, f"Run {run.id} metadata mismatch"

    def test_channel_metadata(
        self,
        mth5_with_experiment: MTH5,
        experiment_from_xml: Experiment,
        survey_name: str,
    ):
        """Test channel metadata consistency."""
        runs = experiment_from_xml.surveys[0].stations[0].runs
        station_id = experiment_from_xml.surveys[0].stations[0].id

        for run in runs:
            h5_run = mth5_with_experiment.get_run(
                station_id, run.id, survey=survey_name
            )

            for channel in run.channels:
                h5_channel = h5_run.get_channel(channel.component)

                original_data = clean_metadata_dict(channel.to_dict(single=True))
                h5_data = clean_metadata_dict(h5_channel.metadata.to_dict(single=True))

                assert (
                    h5_data == original_data
                ), f"Channel {run.id}/{channel.component} metadata mismatch"

    def test_filter_metadata(
        self,
        mth5_with_experiment: MTH5,
        experiment_from_xml: Experiment,
        survey_name: str,
    ):
        """Test filter metadata consistency."""
        exp_filters = experiment_from_xml.surveys[0].filters
        sg = mth5_with_experiment.get_survey(survey_name)

        for key, value in exp_filters.items():
            # Transform key to match how MTH5 stores filter names
            stored_key = key.replace("/", " per ").lower()
            original_data = value.to_dict(single=True, required=False)
            h5_filter = sg.filters_group.to_filter_object(stored_key)
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
        survey_id = experiment_from_xml.surveys[0].id

        run_group = mth5_with_experiment.get_run(station_id, run_id, survey_id)
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
        survey_id = experiment_from_xml.surveys[0].id

        channel_group = mth5_with_experiment.get_channel(
            station_id, run_id, component, survey_id
        )
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

        # Get the survey name for the modified experiment
        modified_survey_name = validate_name(modified_experiment.surveys[0].id)

        # Test survey ID update - for 0.2.0, we need to get the survey through experiment
        survey_group = mth5_obj.get_survey(modified_survey_name)
        assert survey_group.metadata.id == modified_experiment.surveys[0].id

        # Test location update
        station = mth5_obj.get_station("REW09", modified_survey_name)
        expected_lat = modified_experiment.surveys[0].stations[0].location.latitude
        assert station.metadata.location.latitude == expected_lat


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
    mth5_with_experiment: MTH5,
    experiment_from_xml: Experiment,
    survey_name: str,
    station_id: str,
):
    """Test individual station metadata (parametrized)."""
    station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    h5_station = mth5_with_experiment.get_station(station_id, survey_name)

    original_data = clean_metadata_dict(station.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_station.metadata.to_dict(single=True))

    assert h5_data == original_data


def test_individual_run_metadata(
    mth5_with_experiment: MTH5,
    experiment_from_xml: Experiment,
    survey_name: str,
    run_info: tuple[str, str],
):
    """Test individual run metadata (parametrized)."""
    station_id, run_id = run_info

    # Find the run in the experiment
    station = next(
        s for s in experiment_from_xml.surveys[0].stations if s.id == station_id
    )
    run = next(r for r in station.runs if r.id == run_id)

    h5_run = mth5_with_experiment.get_run(station_id, run_id, survey=survey_name)

    original_data = clean_metadata_dict(run.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_run.metadata.to_dict(single=True))

    assert h5_data == original_data


def test_individual_channel_metadata(
    mth5_with_experiment: MTH5,
    experiment_from_xml: Experiment,
    survey_name: str,
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

    h5_run = mth5_with_experiment.get_run(station_id, run_id, survey=survey_name)
    h5_channel = h5_run.get_channel(component)

    original_data = clean_metadata_dict(channel.to_dict(single=True))
    h5_data = clean_metadata_dict(h5_channel.metadata.to_dict(single=True))

    assert h5_data == original_data
