#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mt_io.nims import NIMSCollection

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class NIMSClient(ClientBase):
    def __init__(
        self,
        data_path: str | Path,
        sample_rates: list[int] = [1, 8],
        save_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        mth5_filename: str = "from_nims.h5",
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
        self.collection = NIMSCollection(self.data_path)

    @property
    def calibration_path(self) -> Path | None:
        """
        Path to calibration data.

        Returns
        -------
        Path or None
            Path to calibration file, or None if not set.

        Examples
        --------
        >>> client = NIMSClient('data_dir')
        >>> client.calibration_path = 'calib.dat'
        >>> print(client.calibration_path)
        PosixPath('calib.dat')
        """
        return self._calibration_path

    @calibration_path.setter
    def calibration_path(self, value: str | Path | None) -> None:
        """
        Set the calibration path.

        Parameters
        ----------
        value : str or Path or None
            Path to calibration file, or None.

        Raises
        ------
        IOError
            If the path does not exist.

        Examples
        --------
        >>> client = NIMSClient('data_dir')
        >>> client.calibration_path = 'calib.dat'
        """
        if value is not None:
            self._calibration_path = Path(value)
            if not self._calibration_path.exists():
                raise IOError(f"Could not find {self._calibration_path}")
        else:
            self._calibration_path = None

    def get_run_dict(self) -> dict:
        """
        Get run information from the NIMS collection.

        Returns
        -------
        dict
            Dictionary of run information.

        Examples
        --------
        >>> client = NIMSClient('data_dir')
        >>> runs = client.get_run_dict()
        >>> print(list(runs.keys()))
        ['station1', 'station2']
        """
        return self.collection.get_runs(
            sample_rates=self.sample_rates,
            calibration_path=self.calibration_path,
        )

    def get_survey(self, station_dict: dict) -> str:
        """
        Get survey name from a dictionary of a single station of runs.

        Parameters
        ----------
        station_dict : dict
            Dictionary of runs for a station.

        Returns
        -------
        str
            Survey name.

        Examples
        --------
        >>> client = NIMSClient('data_dir')
        >>> runs = client.get_run_dict()
        >>> survey = client.get_survey(runs['station1'])
        >>> print(survey)
        'survey_name'
        """
        return list(
            set([station_dict[k].survey.unique()[0] for k in station_dict.keys()])
        )[0]

    def make_mth5_from_nims(
        self,
        survey_id: str = "default_survey",
        combine: bool = True,
        **kwargs,
    ) -> str | Path:
        """
        Make an MTH5 file from Phoenix NIMS files. Splits into runs, accounts for filters.

        Parameters
        ----------
        survey_id : str, optional
            Survey identifier. Default is "default_survey".
        combine : bool, optional
            Whether to combine runs. Default is True.
        **kwargs
            Additional keyword arguments to set as attributes.

        Returns
        -------
        str or Path
            Path to the saved MTH5 file.

        Examples
        --------
        >>> client = NIMSClient('data_dir')
        >>> mth5_path = client.make_mth5_from_nims(survey_id='survey1')
        >>> print(mth5_path)
        'output_dir/from_nims.h5'
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        if survey_id is None:
            self.collection.survey_id = "none"
        else:
            self.collection.survey_id = survey_id

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, self.mth5_file_mode)
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(station_id)
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    # use iloc to avoid KeyError if the DataFrame index does not start at 0
                    run_ts = read_file(run_df["fn"].iloc[0])
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path
