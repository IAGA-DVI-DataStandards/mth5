# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:23:50 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from mt_io.metronix import MetronixCollection

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class MetronixClient(ClientBase):
    def __init__(
        self,
        data_path,
        sample_rates=[128],
        save_path=None,
        calibration_path=None,
        mth5_filename="from_metronix.h5",
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
        self.collection = MetronixCollection(self.data_path)

    def get_run_dict(self, run_name_zeros=0):
        """
        get run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            run_name_zeros=run_name_zeros,
            calibration_path=self.calibration_path,
        )

    def get_survey_id(self, station_dict):
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

    def set_station_metadata(self, station_dict, station_group):
        """
        set station group metadata from information in the station dict

        :param station_dict: DESCRIPTION
        :type station_dict: TYPE
        :param station_group: DESCRIPTION
        :type station_group: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        runs = pd.concat(station_dict.values())
        station_group.metadata.location.latitude = runs.latitude.mean()
        station_group.metadata.location.longitude = runs.longitude.mean()
        station_group.metadata.location.elevation = runs.elevation.mean()
        station_group.write_metadata()

    def make_mth5_from_metronix(self, run_name_zeros=0, **kwargs):
        """
        Create an MTH5 from new ATSS + JSON style Metronix data.

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        runs = self.get_run_dict(run_name_zeros=run_name_zeros)

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, self.mth5_file_mode)

            for station_id, station_dict in runs.items():
                survey_id = self.get_survey_id(station_dict)
                survey_group = m.add_survey(survey_id)
                station_group = survey_group.stations_group.add_station(station_id)
                self.set_station_metadata(station_dict, station_group)

                for run_id, run_df in station_dict.items():
                    run_group = station_group.add_run(run_id)
                    for row in run_df.itertuples():
                        ch_ts = read_file(row.fn)
                        run_group.from_channel_ts(ch_ts)
                    run_group.update_metadata()
                station_group.update_metadata()
                survey_group.update_metadata

        self.logger.info(f"Wrote MTH5 file to: {self.save_path}")

        return self.save_path
