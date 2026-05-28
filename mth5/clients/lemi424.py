# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:57:54 2024

@author: jpeacock
"""

from pathlib import Path
from typing import Any, Optional, Union

from mt_io.lemi import LEMICollection

from mth5 import read_file
from mth5.clients.base import ClientBase

# =============================================================================
# Imports
# =============================================================================
from mth5.mth5 import MTH5


# =============================================================================


class LEMI424Client(ClientBase):
    def __init__(
        self,
        data_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        mth5_filename: str = "from_lemi424.h5",
        **kwargs: Any,
    ) -> None:
        """
        LEMI 424 client for converting long period data to MTH5.

        Parameters
        ----------
        data_path : str or Path
            Directory where LEMI 424 data files are located.
        save_path : str or Path, optional
            Directory to save the mth5 file. If None, uses data_path.
        mth5_filename : str, optional
            Name of the mth5 file to create. Default is 'from_lemi424.h5'.
        **kwargs : Any
            Additional keyword arguments for h5 parameters.

        Examples
        --------
        >>> client = LEMI424Client(data_path="./data", save_path="./output")
        >>> client.save_path
        PosixPath('output/from_lemi424.h5')
        """
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=[1],
            mth5_filename=mth5_filename,
            **kwargs,
        )
        self.collection = LEMICollection(self.data_path)

    def make_mth5_from_lemi424(
        self,
        survey_id: str,
        station_id: str,
        **kwargs: Any,
    ) -> Path:
        """
        Create an MTH5 file from LEMI 424 long period data.

        Parameters
        ----------
        survey_id : str
            Survey identifier.
        station_id : str
            Station identifier.
        **kwargs : Any
            Additional keyword arguments to set as attributes.

        Returns
        -------
        Path
            Path to the created mth5 file.

        Examples
        --------
        >>> client = LEMI424Client(data_path="./data")
        >>> client.make_mth5_from_lemi424("SURVEY1", "ST01")
        PosixPath('data/from_lemi424.h5')
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        self.collection.survey_id = survey_id
        self.collection.station_id = station_id

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, self.mth5_file_mode)
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(station_id)
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_file(
                        run_df.fn.to_list(),
                        calibration_dict=self.collection.calibration_dict,
                    )
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path
