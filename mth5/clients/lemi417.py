# -*- coding: utf-8 -*-
"""
LEMI 417 Client
================

Client for converting LEMI 417 long period data into MTH5 format.
@author: xubc
Implemented based on lemi424.py(by jpeacock).
"""

from pathlib import Path
from typing import Any, Optional, Union

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.io.lemi417 import LEMICollection

# =============================================================================
# Imports
# =============================================================================
from mth5.mth5 import MTH5


# =============================================================================


class LEMI417Client(ClientBase):
    def __init__(
        self,
        data_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        mth5_filename: str = "from_lemi417.h5",
        **kwargs: Any,
    ) -> None:
        """
        LEMI 417 client for converting long period data to MTH5.

        Parameters
        ----------
        data_path : str or Path
            Directory containing LEMI 417 data files.
        save_path : str or Path, optional
            Path to save the MTH5 file. If None, uses data_path.
        mth5_filename : str, optional
            MTH5 output filename, default is 'from_lemi417.h5'.
        **kwargs : Any
            Additional keyword arguments for HDF5 configuration.
        """
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=[1],
            mth5_filename=mth5_filename,
            **kwargs,
        )
        self.collection = LEMICollection(self.data_path)

    def make_mth5_from_lemi417(
        self,
        survey_id: str,
        station_id: str,
        **kwargs: Any,
    ) -> Path:
        """
        Create an MTH5 file from LEMI 417 long period data.

        Parameters
        ----------
        survey_id : str
            Survey identifier.
        station_id : str
            Station identifier.
        **kwargs : Any
            Additional attribute parameters.

        Returns
        -------
        Path
            Path to the generated MTH5 file.
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
                        file_type="lemi417",
                        calibration_dict=self.collection.calibration_dict,
                    )
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # Update survey metadata
            survey_group.update_metadata()

        return self.save_path
