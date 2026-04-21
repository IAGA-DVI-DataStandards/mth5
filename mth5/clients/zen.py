#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mt_io.zen import Z3DCollection

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class ZenClient(ClientBase):
    def __init__(
        self,
        data_path,
        sample_rates=[4096, 1024, 256],
        save_path=None,
        calibration_path=None,
        mth5_filename="from_zen.h5",
        **kwargs,
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.calibration_path = calibration_path
        self.collection = Z3DCollection(self.data_path)
        self.station_stem = None

    @property
    def calibration_path(self):
        """Path to calibration data"""
        return self._calibration_path

    @calibration_path.setter
    def calibration_path(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if value is not None:
            self._calibration_path = Path(value)
            if not self._calibration_path.exists():
                raise IOError(f"Could not find {self._calibration_path}")

        else:
            self._calibration_path = None
            # raise ValueError("calibration_path cannot be None")

    def get_run_dict(self):
        """
        Get Run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            calibration_path=self.calibration_path,
        )

    def get_survey(self, station_dict):
        """
        get survey name from a dictionary of a single station of runs
        :param station_dict: DESCRIPTION
        :type station_dict: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return list(
            set([station_dict[k].survey.unique()[0] for k in station_dict.keys()])
        )[0]

    def _get_station_group_id(self, station_id):
        """
        Get the station group id from the station id.
        :param station_id: DESCRIPTION
        :type station_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if self.station_stem is not None:
            if not station_id.startswith(self.station_stem):
                return f"{self.station_stem}{station_id}"
            else:
                return station_id
        else:
            return station_id

    def make_mth5_from_zen(self, survey_id=None, combine=True, **kwargs):
        """
        Make an MTH5 from Phoenix files.  Split into runs, account for filters

        :param data_path: DESCRIPTION, defaults to None
        :type data_path: TYPE, optional
        :param sample_rates: DESCRIPTION, defaults to None
        :type sample_rates: TYPE, optional
        :param save_path: DESCRIPTION, defaults to None
        :type save_path: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, self.mth5_file_mode)

            for station_id, station_dict in runs.items():
                if survey_id is None:
                    survey_id = self.get_survey(station_dict)
                survey_group = m.add_survey(survey_id)
                station_group_id = self._get_station_group_id(station_id)
                station_group = survey_group.stations_group.add_station(
                    station_group_id
                )
                station_group.metadata.update(
                    self.collection.station_metadata_dict[station_id]
                )
                station_group.metadata.id = station_group_id
                station_group.write_metadata()
                if combine:
                    run_list = []
                for run_id, run_df in station_dict.items():
                    run_group = station_group.add_run(run_id)
                    for row in run_df.itertuples():
                        ch_ts = read_file(
                            row.fn,
                            calibration_fn=row.calibration_fn,
                        )
                        run_group.from_channel_ts(ch_ts)
                    run_group.update_metadata()
                    if combine:
                        run_list.append(run_group.to_runts())
                if combine:
                    # Combine runs and down sample to 1 second.
                    combined_run = run_list[0].merge(run_list[1:], new_sample_rate=1)
                    combined_run.run_metadata.id = "sr1_0001"
                    combined_run_group = station_group.add_run("sr1_0001")
                    combined_run_group.from_runts(combined_run)
                    combined_run_group.update_metadata()
                station_group.update_metadata()
            survey_group.update_metadata()

        self.logger.info(f"Wrote MTH5 file to: {self.save_path}")

        return self.save_path
