# -*- coding: utf-8 -*-
"""Pytest suite for LEMI417Client using mock-driven, xdist-safe tests."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from mt_io.lemi import LEMICollection

from mth5.clients.lemi417 import LEMI417Client


@pytest.fixture
def data_dir(tmp_path):
    """Per-test data directory (xdist-safe)."""
    d = tmp_path / "lemi417_data"
    d.mkdir()
    return d


@pytest.fixture
def basic_client(data_dir):
    """Basic LEMI417 client instance."""
    return LEMI417Client(data_dir, h5_mode="w", h5_driver="sec2")


@pytest.fixture
def mock_collection():
    """Mock LEMICollection with a single station/run."""
    collection = Mock(spec=LEMICollection)
    collection.survey_id = "survey_a"
    collection.station_id = "station_a"
    collection.calibration_dict = {"bx": "bx_cal.json"}
    collection.get_runs.return_value = {
        "station_a": {
            "sr1_001": Mock(fn=Mock(to_list=Mock(return_value=["f1.bin", "f2.bin"])))
        }
    }
    return collection


class TestLEMI417ClientInitialization:
    @pytest.mark.parametrize("as_string", [True, False])
    def test_init_defaults(self, data_dir, as_string):
        input_path = str(data_dir) if as_string else data_dir
        client = LEMI417Client(input_path)

        assert client.data_path == data_dir
        assert client.sample_rates == [1]
        assert client.mth5_filename == "from_lemi417.h5"
        assert client.save_path == data_dir / "from_lemi417.h5"
        assert isinstance(client.collection, LEMICollection)

    def test_init_custom_save_path_and_filename(self, data_dir, tmp_path):
        save_file = tmp_path / "custom_lemi417.h5"
        client = LEMI417Client(data_dir, save_path=save_file)

        assert client.save_path == save_file
        assert client.mth5_filename == "custom_lemi417.h5"

    def test_init_none_data_path_raises(self):
        with pytest.raises(ValueError, match="data_path cannot be None"):
            LEMI417Client(None)

    def test_init_missing_data_path_raises(self):
        with pytest.raises(IOError, match="Could not find"):
            LEMI417Client(Path("missing_dir_12345"))


class TestLEMI417ClientRunCollectionIntegration:
    def test_get_run_dict_uses_collection_with_sample_rates(self, basic_client):
        with patch.object(basic_client.collection, "get_runs") as mock_get_runs:
            mock_get_runs.return_value = {"ok": "value"}
            basic_client.sample_rates = [1, 4]

            out = basic_client.get_run_dict()

        assert out == {"ok": "value"}
        mock_get_runs.assert_called_once_with(sample_rates=[1.0, 4.0])


class TestLEMI417ClientMTH5Creation:
    @patch("mth5.clients.lemi417.read_file")
    @patch("mth5.clients.lemi417.MTH5")
    def test_make_mth5_from_lemi417_happy_path(
        self,
        mock_mth5_class,
        mock_read_file,
        basic_client,
        mock_collection,
        subtests,
    ):
        mock_mth5 = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5

        mock_survey_group = MagicMock()
        mock_station_group = MagicMock()
        mock_run_group = MagicMock()

        mock_mth5.add_survey.return_value = mock_survey_group
        mock_survey_group.stations_group.add_station.return_value = mock_station_group
        mock_station_group.add_run.return_value = mock_run_group

        mock_run_ts = MagicMock()
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        basic_client.collection = mock_collection

        out = basic_client.make_mth5_from_lemi417("survey_in", "station_in")

        with subtests.test("return value"):
            assert out == basic_client.save_path

        with subtests.test("survey and station are propagated"):
            assert basic_client.collection.survey_id == "survey_in"
            assert basic_client.collection.station_id == "station_in"

        with subtests.test("open and survey creation"):
            mock_mth5.open_mth5.assert_called_once_with(basic_client.save_path, "w")
            mock_mth5.add_survey.assert_called_once_with("survey_in")

        with subtests.test("reader call includes file type and calibration"):
            mock_read_file.assert_called_once_with(
                ["f1.bin", "f2.bin"],
                file_type="lemi417",
                calibration_dict=basic_client.collection.calibration_dict,
            )

        with subtests.test("run write path"):
            mock_station_group.add_run.assert_called_once_with("sr1_001")
            assert mock_run_ts.run_metadata.id == "sr1_001"
            mock_run_group.from_runts.assert_called_once_with(mock_run_ts)

        with subtests.test("metadata update and flush"):
            mock_station_group.metadata.update.assert_called_once_with(
                mock_run_ts.station_metadata
            )
            mock_station_group.write_metadata.assert_called_once_with()
            mock_survey_group.update_metadata.assert_called_once_with()

    @pytest.mark.parametrize(
        "kwargs_in, expected_mode",
        [
            ({"mth5_file_mode": "a"}, "a"),
            ({"mth5_file_mode": None}, "w"),
        ],
    )
    @patch("mth5.clients.lemi417.read_file")
    @patch("mth5.clients.lemi417.MTH5")
    def test_make_mth5_kwargs_update_attributes(
        self,
        mock_mth5_class,
        mock_read_file,
        kwargs_in,
        expected_mode,
        basic_client,
        mock_collection,
    ):
        mock_mth5 = MagicMock()
        mock_mth5_class.return_value.__enter__.return_value = mock_mth5
        mock_survey_group = MagicMock()
        mock_station_group = MagicMock()

        mock_mth5.add_survey.return_value = mock_survey_group
        mock_survey_group.stations_group.add_station.return_value = mock_station_group
        mock_station_group.add_run.return_value = MagicMock()

        mock_run_ts = MagicMock()
        mock_run_ts.station_metadata = MagicMock()
        mock_read_file.return_value = mock_run_ts

        basic_client.collection = mock_collection

        basic_client.make_mth5_from_lemi417("survey_in", "station_in", **kwargs_in)

        mock_mth5.open_mth5.assert_called_once_with(
            basic_client.save_path, expected_mode
        )
