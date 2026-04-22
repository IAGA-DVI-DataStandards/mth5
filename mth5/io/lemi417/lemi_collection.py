# -*- coding: utf-8 -*-
"""
LEMI 417 Collection
====================

Collection of LEMI 417 files combined into runs

Created on Wed Mar 05 2026

@author: xubc
Modified from "lemi_collection.py" (by Jared Peacock)
"""

import pathlib
from pathlib import Path
from typing import List

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5.io.collection import Collection
from mth5.io.lemi417 import LEMI417


# =============================================================================


class LEMICollection(Collection):
    """
    Collection of LEMI 417 files into runs based on start and end times.

    Will assign the run name as 'sr1_{index:0{zeros}}' --> 'sr1_0001' for
    `zeros` = 4.

    Notes
    -----
    This class assumes that the given file path contains a single
    LEMI station. If you want to do multiple stations merge the returned
    data frames.

    LEMI data comes with little metadata about the station or survey,
    therefore you should assign `station_id` and `survey_id`.

    Parameters
    ----------
    file_path : str or pathlib.Path, optional
        Full path to single station LEMI417 directory, by default None
    file_ext : list of str, optional
        Extension of LEMI417 files, format is "B+instrument_number"
        (e.g. "B16", "B05"), by default ["B*"] to match all LEMI417 files
    **kwargs
        Additional keyword arguments passed to parent Collection class

    Attributes
    ----------
    station_id : str
        Station identification string, defaults to "mt001"
    survey_id : str
        Survey identification string, defaults to "mt"

    Examples
    --------
    >>> from mth5.io.lemi417 import LEMICollection
    >>> lc = LEMICollection(r"/path/to/single/lemi417/station")
    >>> lc.station_id = "mt001"
    >>> lc.survey_id = "test_survey"
    >>> run_dict = lc.get_runs(1)
    """

    def __init__(
        self,
        file_path: str | pathlib.Path | None = None,
        file_ext: List[str] | None = None,
        **kwargs,
    ) -> None:
        if file_ext is None:
            file_ext = ["B*"]
        super().__init__(file_path=file_path, file_ext=file_ext, **kwargs)

        self.station_id = "mt001"
        self.survey_id = "mt"
        self.calibration_dict = {}

    def get_calibrations(self, calibration_path: str | Path) -> dict:
        """
        Get calibration dictionary for LEMI417 files.  This assumes that the
        calibration files are in JSON format and named as
        'LEMI-417-<component>.json'

        Parameters
        ----------
        calibration_path : str or pathlib.Path
            Path to calibration files

        Returns
        -------
        dict
            Calibration dictionary for LEMI417 files

        Examples
        --------
        >>> from mth5.io.lemi417 import LEMICollection
        >>> lc = LEMICollection("/path/to/single/lemi417/station")
        >>> cal_dict = lc.get_calibrations(Path("/path/to/calibrations"))
        """
        calibration_path = Path(calibration_path)

        calibration_dict = {}
        for fn in calibration_path.rglob("*.json"):
            comp = fn.stem.split("-")[-1].split(".", 1)[0]
            calibration_dict[comp] = fn

        return calibration_dict

    def to_dataframe(
        self,
        sample_rates: int | List[int] | None = None,
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Create a data frame of each LEMI417 file in a given directory.

        Notes
        -----
        This assumes the given directory contains a single station

        Parameters
        ----------
        sample_rates : int or list of int, optional
            Sample rate to get, will always be 1 for LEMI data, by default [1]
        run_name_zeros : int, optional
            Number of zeros to assign to the run name, by default 4
        calibration_path : str or pathlib.Path, optional
            Path to calibration files, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame with information of each LEMI417 file in the given directory

        Examples
        --------
        >>> from mth5.io.lemi417 import LEMICollection
        >>> lc = LEMICollection("/path/to/single/lemi417/station")
        >>> lemi_df = lc.to_dataframe()
        """
        if sample_rates is None:
            sample_rates = [1]

        if calibration_path is None:
            calibration_path = Path(self.file_path)
        self.calibration_dict = self.get_calibrations(calibration_path)
        if not self.calibration_dict:
            self.logger.warning(
                f"No calibration files found in {calibration_path}, "
                "proceeding without calibrations."
            )

        entries = []
        for fn in self.get_files(self.file_ext):
            lemi_obj = LEMI417(fn)
            lemi_obj.read()
            n_samples = int(lemi_obj.n_samples or 0)

            entry = self.get_empty_entry_dict()
            entry["survey"] = self.survey_id
            entry["station"] = self.station_id
            entry["start"] = lemi_obj.start.isoformat() if lemi_obj.start else ""
            entry["end"] = lemi_obj.end.isoformat() if lemi_obj.end else ""
            entry["component"] = ",".join(lemi_obj.run_metadata.channels_recorded_all)
            entry["fn"] = fn
            entry["sample_rate"] = lemi_obj.sample_rate
            entry["file_size"] = lemi_obj.file_size
            entry["n_samples"] = n_samples

            entries.append(entry)

        # make pandas dataframe and set data types
        if len(entries) == 0:
            self.logger.warning("No entries found for LEMI-417 collection")
            return pd.DataFrame()

        df = pd.DataFrame(entries)
        df.loc[:, "channel_id"] = 1
        df.loc[:, "sequence_number"] = 0
        df.loc[:, "instrument_id"] = "LEMI417"

        df = self._sort_df(self._set_df_dtypes(df), run_name_zeros)

        return df

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 4) -> pd.DataFrame:
        """
        Assign run names based on start and end times.

        Checks if a file has the same start time as the last end time.
        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the appropriate columns
        zeros : int, optional
            Number of zeros in run name, by default 4

        Returns
        -------
        pd.DataFrame
            DataFrame with run names assigned
        """
        count = 1
        for row in df.itertuples():
            if row.Index == 0:
                df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                previous_end = row.end
            else:
                if (
                    row.start - previous_end
                ).total_seconds() / row.sample_rate == row.sample_rate:
                    df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                else:
                    count += 1
                    df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
                previous_end = row.end

        return df
