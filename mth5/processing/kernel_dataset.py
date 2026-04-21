"""
Magnetotelluric kernel dataset processing module.

This module contains a class for representing a dataset that can be processed.

The module provides functionality for:
- Managing magnetotelluric time series intervals
- Supporting single station and remote reference processing
- Handling run combination and time interval restrictions
- Interfacing with MTH5 data structures

Development Notes
-----------------
Players on the stage: One or more mth5s.

Each mth5 has a "run_summary" dataframe available. Run_summary provides options for
the local and possibly remote reference stations. Candidates for local station are
the unique values in the station column.

For any candidate station, there are some integer n runs available.
This yields 2^n - 1 possible combinations that can be processed, neglecting any
flagging of time intervals within any run, or any joining of runs.
(There are actually 2**n, but we ignore the empty set, so -1)

Intuition suggests default ought to be to process n runs in n+1 configurations:
{all runs} + each run individually. This will give a bulk answer, and bad runs can
be flagged by comparing them. After an initial processing, the tfs can be reviewed
and the problematic runs can be addressed.

The user can interact with the run_summary_df, selecting sub dataframes via querying,
and in future maybe via some GUI (or a spreadsheet).

The intended usage process is as follows:
 0. Start with a list of mth5s
 1. Extract a run_summary
 2. Stare at the run_summary_df, and select a station "S" to process
 3. Select a non-empty set of runs for station "S"
 4. Select a remote reference "RR", (this is allowed to be None)
 5. Extract the sub-dataframe corresponding to acquisition_runs from "S" and "RR"
 7. If the remote is not None:
  - Drop the runs (rows) associated with RR that do not intersect with S
  - Restrict start/end times of RR runs that intersect with S so overlap is complete.
  - Restrict start/end times of S runs so that they intersect with remote
 8. This is now a TFKernel Dataset Definition (ish). Initialize a default processing
 object and pass it this df.

Examples
--------
>>> cc = ConfigCreator()
>>> p = cc.create_from_kernel_dataset(kernel_dataset)
- Optionally pass emtf_band_file=emtf_band_setup_file
 9. Edit the Processing Config appropriately,

TODO: Consider supporting a default value for 'channel_scale_factors' that is None,
TODO: Might need to groupby survey & station, for now consider station_id unique.
"""

from __future__ import annotations

import copy

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import mt_metadata.timeseries
import mt_timeseries.run_ts
import pandas as pd
from loguru import logger
from mt_metadata.common.list_dict import ListDict
from mt_metadata.timeseries import Survey
from mt_metadata.transfer_functions.tf import Station

from mth5.mth5 import MTH5
from mth5.processing import KERNEL_DATASET_DTYPE, MINI_SUMMARY_COLUMNS
from mth5.processing.run_summary import RunSummary
from mth5.utils.helpers import initialize_mth5


# =============================================================================


class KernelDataset:
    """
    Magnetotelluric kernel dataset for time series processing.

    This class works with mth5-derived channel_summary or run_summary dataframes
    that specify time series intervals. It manages acquisition "runs" that can be
    merged into processing runs, with support for both single station and remote
    reference processing configurations.

    Parameters
    ----------
    df : pd.DataFrame | None, optional
        Pre-formed dataframe with dataset configuration. Normally built from a
        run_summary, by default None
    local_station_id : str, optional
        Local station identifier for the dataset. Normally passed via
        from_run_summary method, by default ""
    remote_station_id : str | None, optional
        Remote reference station identifier. Normally passed via from_run_summary
        method, by default None
    **kwargs : dict
        Additional keyword arguments to set as attributes

    Attributes
    ----------
    df : pd.DataFrame | None
        Main dataset dataframe with time series intervals
    local_station_id : str | None
        Local station identifier
    remote_station_id : str | None
        Remote reference station identifier
    survey_metadata : dict
        Survey metadata container
    initialized : bool
        Whether MTH5 objects have been initialized
    local_mth5_obj : Any | None
        Local station MTH5 object
    remote_mth5_obj : Any | None
        Remote station MTH5 object

    Notes
    -----
    The class is closely related to (may actually be an extension of) RunSummary.
    The main idea is to specify one or two stations, and a list of acquisition "runs"
    that can be merged into a "processing run". Each acquisition run can be further
    divided into non-overlapping chunks by specifying time-intervals associated with
    that acquisition run.

    The time intervals can be used for several purposes but primarily:
    - STFT processing for merged FC data structures
    - Binding together into xarray time series for gap filling
    - Managing and analyzing availability of reference time series

    Examples
    --------
    Create a kernel dataset from run summary:

    >>> from mth5.processing.run_summary import RunSummary
    >>> run_summary = RunSummary()
    >>> dataset = KernelDataset()
    >>> dataset.from_run_summary(run_summary, "station01", "station02")

    Process single station data:

    >>> single_dataset = KernelDataset()
    >>> single_dataset.from_run_summary(run_summary, "station01")

    See Also
    --------
    RunSummary : Data summary for processing configuration
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        local_station_id: str = "",
        remote_station_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize KernelDataset instance.

        Parameters
        ----------
        df : pd.DataFrame | None, optional
            Pre-formed dataframe with dataset configuration, by default None
        local_station_id : str, optional
            Local station identifier, by default ""
        remote_station_id : str | None, optional
            Remote reference station identifier, by default None
        **kwargs : dict
            Additional keyword arguments to set as attributes
        """
        self.df = df
        self.local_station_id = local_station_id
        self.remote_station_id = remote_station_id
        self._mini_summary_columns = MINI_SUMMARY_COLUMNS
        self.survey_metadata: dict[str, Any] = {}
        self.initialized: bool = False
        self.local_mth5_obj: Any = None
        self.remote_mth5_obj: Any = None
        self._local_mth5_path: Path | None = None
        self._remote_mth5_path: Path | None = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        """
        Return string representation of the dataset.

        Returns
        -------
        str
            String representation showing mini summary
        """
        return str(self.mini_summary)

    def __repr__(self) -> str:
        """
        Return detailed string representation.

        Returns
        -------
        str
            Detailed string representation
        """
        return self.__str__()

    @property
    def df(self) -> pd.DataFrame | None:
        """
        Main dataset dataframe.

        Returns
        -------
        pd.DataFrame | None
            Dataset dataframe with time series intervals, or None if not set
        """
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame | None) -> None:
        """
        Set the dataset dataframe with proper validation.

        Parameters
        ----------
        value : pd.DataFrame | None
            Dataframe to set, or None to clear

        Raises
        ------
        TypeError
            If value is not a DataFrame or None
        """
        if value is None:
            self._df = None
            return

        if not isinstance(value, pd.DataFrame):
            msg = f"Need to set df with a Pandas.DataFrame not type({type(value)})"
            logger.error(msg)
            raise TypeError(msg)

        self._df = self._add_duration_column(
            self._set_datetime_columns(self._add_columns(value)), inplace=False
        )

    def _has_df(self) -> bool:
        """
        Check if dataframe is set and not empty.

        Returns
        -------
        bool
            True if dataframe exists and is not empty
        """
        if self._df is not None:
            if not self._df.empty:
                return True
            return False
        return False

    def _df_has_local_station_id(self, df: pd.DataFrame) -> bool:
        """
        Check if dataframe contains the local station ID.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to check

        Returns
        -------
        bool
            True if local station ID exists in dataframe
        """
        return (df.station == self.local_station_id).any()

    def _df_has_remote_station_id(self, df: pd.DataFrame) -> bool:
        """
        Check if dataframe contains the remote station ID.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to check

        Returns
        -------
        bool
            True if remote station ID exists in dataframe
        """
        return (df.station == self.remote_station_id).any()

    def _set_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure start and end columns are datetime objects.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with datetime columns properly set
        """
        try:
            df.start = pd.to_datetime(df.start, format="mixed")
            df.end = pd.to_datetime(df.end, format="mixed")
        except ValueError:
            df.start = pd.to_datetime(df.start)
            df.end = pd.to_datetime(df.end)

        return df

    def clone(self) -> "KernelDataset":
        """
        Create a deep copy of the dataset.

        Returns
        -------
        KernelDataset
            Deep copy of this instance
        """
        return copy.deepcopy(self)

    def clone_dataframe(self) -> pd.DataFrame | None:
        """
        Create a deep copy of the dataframe.

        Returns
        -------
        pd.DataFrame | None
            Deep copy of the dataframe, or None if dataframe is not set
        """
        return copy.deepcopy(self.df)

    def _add_columns(
        self,
        df: pd.DataFrame,
        null_columns: list[str] | tuple[str, ...] = ("fc",),
    ) -> pd.DataFrame:
        """
        Add missing columns with appropriate dtypes.

        Parameters
        ----------
        df : pd.DataFrame
            Kernel dataset dataframe, possibly missing some columns
        null_columns : list[str] | tuple[str, ...], optional
            Columns that will initialize to null rather than their expected dtype,
            by default ("fc",)

        Returns
        -------
        pd.DataFrame
            Kernel dataset dataframe with all required columns present

        Raises
        ------
        ValueError
            If required columns (survey, station, run, start, end) are missing
        """
        for col, dtype in KERNEL_DATASET_DTYPE:
            if col not in df.columns:
                if col in ["survey", "station", "run", "start", "end"]:
                    raise ValueError(f"{col} must be a filled column in the dataframe")

                try:
                    df[col] = dtype(0)
                    assigned_dtype = dtype
                except TypeError:
                    df[col] = None  # TODO: update to pd.NA
                    assigned_dtype = type(None)

                if col in null_columns:
                    df[col] = pd.NA
                    assigned_dtype = type(pd.NA)

                msg = (
                    f"KernelDataset DataFrame needs column {col}, adding and "
                    f"setting dtype to {assigned_dtype}."
                )
                logger.debug(msg)

        return df

    @property
    def local_station_id(self) -> str | None:
        """
        Local station identifier.

        Returns
        -------
        str | None
            Local station identifier
        """
        return self._local_station_id

    @local_station_id.setter
    def local_station_id(self, value: str | None) -> None:
        """
        Set local station identifier.

        Parameters
        ----------
        value : str | None
            Station identifier to set

        Raises
        ------
        ValueError
            If value cannot be converted to string
        NameError
            If station ID is not found in dataframe when dataframe exists
        """
        if value is None:
            self._local_station_id = None
        else:
            try:
                self._local_station_id = str(value)
            except ValueError:
                raise ValueError(
                    f"Bad type {type(value)}. "
                    "Cannot convert local_station_id value to string."
                )
            if self._has_df() and self.df is not None:
                if not self._df_has_local_station_id(self.df):
                    raise NameError(
                        f"Could not find {self._local_station_id} in dataframe"
                    )

    @property
    def local_mth5_path(self) -> Path | None:
        """
        Local station MTH5 file path.

        Returns
        -------
        Path | None
            Path to local station MTH5 file, extracted from dataframe or
            stored path, or None if not available
        """
        if self._has_df() and self._df is not None:
            unique_paths = self._df.loc[
                self._df.station == self.local_station_id, "mth5_path"
            ].unique()
            if len(unique_paths) > 0:
                return Path(unique_paths[0])
            return None
        else:
            return self._local_mth5_path

    @local_mth5_path.setter
    def local_mth5_path(self, value: str | Path | None) -> None:
        """
        Set local MTH5 path.

        Parameters
        ----------
        value : str | Path | None
            Path to MTH5 file
        """
        self._local_mth5_path = self.set_path(value)

    def has_local_mth5(self) -> bool:
        """
        Check if local MTH5 file exists.

        Returns
        -------
        bool
            True if local MTH5 file exists on filesystem
        """
        if self.local_mth5_path is None:
            return False
        else:
            return self.local_mth5_path.exists()

    @property
    def remote_station_id(self) -> str | None:
        """
        Remote reference station identifier.

        Returns
        -------
        str | None
            Remote station identifier
        """
        return self._remote_station_id

    @remote_station_id.setter
    def remote_station_id(self, value: str | None) -> None:
        """
        Set remote station identifier.

        Parameters
        ----------
        value : str | None
            Remote station identifier

        Raises
        ------
        ValueError
            If value cannot be converted to string
        NameError
            If station ID is not found in dataframe when dataframe exists
        """
        if value is None:
            self._remote_station_id = None
        else:
            try:
                self._remote_station_id = str(value)
            except ValueError:
                raise ValueError(
                    f"Bad type {type(value)}. "
                    "Cannot convert remote_station_id value to string."
                )
            if self._has_df():
                if not self._df_has_remote_station_id(self.df):
                    raise NameError(
                        f"Could not find {self._remote_station_id} in dataframe"
                    )

    @property
    def remote_mth5_path(self) -> Path:
        """Remote mth5 path.
        :return: Remote station MTH5 path, a property extracted from the dataframe.
        :rtype: Path
        """
        if self._has_df() and self.remote_station_id is not None:
            return Path(
                self._df.loc[
                    self._df.station == self.remote_station_id, "mth5_path"
                ].unique()[0]
            )
        else:
            return self._remote_mth5_path

    @remote_mth5_path.setter
    def remote_mth5_path(self, value: str | Path | None):
        """
        Set the remote mth5 path.

        Parameters
        ----------
        value : str | Path | None
            Path to the remote mth5 file
        """
        self._remote_mth5_path = self.set_path(value)

    def has_remote_mth5(self) -> bool:
        """Test if remote mth5 exists."""
        if self.remote_mth5_path is None:
            return False
        else:
            return self.remote_mth5_path.exists()

    @property
    def processing_id(self) -> str:
        """Its difficult to come up with unique ids without crazy long names
        so this is a generic id of local-remote, the station metadata
        will have run information and the config parameters.
        """
        if self.remote_station_id is not None:
            return (
                f"{self.local_station_id}_rr_{self.remote_station_id}_"
                f"sr{int(self.sample_rate)}"
            )
        else:
            return f"{self.local_station_id}_sr{int(self.sample_rate)}"

    @property
    def input_channels(self) -> list[str]:
        """
        Get input channels from dataframe.

        Returns
        -------
        list[str]
            Input channel identifiers (sources)

        Raises
        ------
        AttributeError
            If dataframe is not available or local_df has no input_channels
        """
        if self._has_df() and self.df is not None:
            local_data = self.local_df
            if local_data is not None and not local_data.empty:
                return local_data.input_channels.iat[0]
        return []

    @property
    def output_channels(self) -> list[str]:
        """
        Get output channels from dataframe.

        Returns
        -------
        list[str]
            Output channel identifiers

        Raises
        ------
        AttributeError
            If dataframe is not available or local_df has no output_channels
        """
        if self._has_df() and self.df is not None:
            local_data = self.local_df
            if local_data is not None and not local_data.empty:
                return local_data.output_channels.iat[0]
        return []

    @property
    def remote_channels(self) -> list[str]:
        """
        Get remote reference channels from dataframe.

        Returns
        -------
        list[str]
            Remote reference channel identifiers

        Raises
        ------
        AttributeError
            If dataframe is not available or remote_df has no remote_channels
        """
        if (
            self._has_df()
            and self.df is not None
            and self.remote_station_id is not None
        ):
            remote_data = self.remote_df
            if remote_data is not None and not remote_data.empty:
                return remote_data.input_channels.iat[0]
        return []

    @property
    def local_df(self) -> pd.DataFrame | None:
        """
        Get dataframe subset for local station runs.

        Returns
        -------
        pd.DataFrame | None
            Local station runs data, or None if dataframe not available
        """
        if self._has_df() and self.df is not None:
            return self.df[self.df.station == self.local_station_id]
        return None

    @property
    def remote_df(self) -> pd.DataFrame | None:
        """
        Get dataframe subset for remote station runs.

        Returns
        -------
        pd.DataFrame | None
            Remote station runs data, or None if dataframe not available
            or no remote station configured
        """
        if (
            self._has_df()
            and self.df is not None
            and self.remote_station_id is not None
        ):
            return self.df[self.df.station == self.remote_station_id]
        return None

    @classmethod
    def set_path(cls, value: str | Path | None) -> Path | None:
        """
        Set and validate a file path.

        Parameters
        ----------
        value : str | Path | None
            Path value to set and validate

        Returns
        -------
        Path | None
            Validated Path object, or None if input is None

        Raises
        ------
        IOError
            If path does not exist on filesystem
        ValueError
            If value cannot be converted to Path
        """
        if value is None:
            return None

        if isinstance(value, (str, Path)):
            return_path = Path(value)
            if not return_path.exists():
                raise IOError(f"Cannot find file: {return_path}")
            return return_path
        else:
            raise ValueError(f"Cannot convert type {type(value)} to Path")

    def from_run_summary(
        self,
        run_summary: RunSummary,
        local_station_id: str | None = None,
        remote_station_id: str | None = None,
        sample_rate: float | int | None = None,
    ) -> None:
        """
        Initialize the dataframe from a run summary.

        Parameters
        ----------
        run_summary : RunSummary
            Summary of available data for processing from one or more stations
        local_station_id : str | None, optional
            Label of the station for which an estimate will be computed,
            by default None
        remote_station_id : str | None, optional
            Label of the remote reference station, by default None
        sample_rate : float | int | None, optional
            Sample rate to filter data by, by default None

        Raises
        ------
        ValueError
            If restricting to specified stations yields empty dataset or
            if local and remote stations do not overlap for remote reference
        """

        self.df = None

        if local_station_id is not None:
            self.local_station_id = local_station_id
        if remote_station_id is not None:
            self.remote_station_id = remote_station_id

        if sample_rate is not None:
            run_summary = run_summary.set_sample_rate(sample_rate)

        station_ids = [local_station_id]
        if self.remote_station_id:
            station_ids.append(remote_station_id)
        # Filter out None values before passing to function
        station_ids_filtered = [sid for sid in station_ids if sid is not None]
        df = restrict_to_station_list(
            run_summary.df, station_ids_filtered, inplace=False
        )

        # Check df is non-empty
        if len(df) == 0:
            msg = f"Restricting run_summary df to {station_ids} yields an empty set"
            logger.critical(msg)
            raise ValueError(msg)

        # Check that the columns have data
        len_df_before_drop_dataless_rows = len(df)
        df = df[df.has_data]
        n_dropped = len_df_before_drop_dataless_rows - len(df)
        if n_dropped:
            msg = f"Dropped {n_dropped} rows from Kernel Dataset due to missing data"
            logger.warning(msg)

        # Check df is non-empty (again)
        if len(df) == 0:
            msg = (
                "Restricting run_summary df to runs that have data yields an empty set"
            )
            logger.critical(msg)
            raise ValueError(msg)

        # add columns column
        df = self._add_columns(df)

        # set remote reference
        if self.remote_station_id:
            cond = df.station == remote_station_id
            df.remote = cond

        # be sure to set date time columns and restrict to simultaneous runs
        df = self._set_datetime_columns(df)
        if self.remote_station_id:
            df = self.restrict_run_intervals_to_simultaneous(df)

        # Again check df is non-empty
        if len(df) == 0:
            msg = (
                f"Local: {local_station_id} and remote: "
                f"{remote_station_id} do not overlap. Remote reference "
                "processing not a valid option."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.df = df

        self.survey_metadata = self.get_metadata_from_df(self.local_df)

    def get_metadata_from_df(self, df: pd.DataFrame) -> Survey:
        """
        Extract metadata from the dataframe.  The data frame should only include one
        station.  So use self.local_df or self.remote_df. (Run Summary)

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to extract metadata from

        Returns
        -------
        dict[str, Any]
            Dictionary containing survey metadata
        """
        if df is None or df.empty:
            return {}

        mth5_path = df["mth5_path"].unique()[0]
        if len(mth5_path) == 0:
            raise ValueError(
                f"Cannot find MTH5 path for local station {self.local_station_id}"
            )

        h5_station_reference = df["station_hdf5_reference"].unique()[0]

        if h5_station_reference is None:
            logger.warning(
                f"Cannot find h5_station_reference for local station {self.local_station_id}"
            )
            return {}
        with MTH5() as m:
            m.open_mth5(mth5_path)
            station_group = m.from_reference(h5_station_reference)
            survey_metadata = station_group.survey_metadata

        # survey metadata returns a time series station, so need to update to a
        # transfer function object

        tf_station = Station()
        tf_station.update(survey_metadata.stations[self.local_station_id])

        # remove runs that are not in the dataframe
        processing_runs = df.run.unique()
        for run in tf_station.runs.keys():
            if run not in processing_runs:
                tf_station.remove_run(run)
            else:
                tf_station.runs[run].update(
                    survey_metadata.stations[self.local_station_id].runs[run]
                )

        # add to survey metadata by removing the old one first
        survey_metadata.remove_station(self.local_station_id)
        survey_metadata.add_station(tf_station)

        return survey_metadata

    @property
    def mini_summary(self) -> pd.DataFrame:
        """
        Return a dataframe that fits in terminal display.

        Returns
        -------
        pd.DataFrame
            Subset of the main dataframe with key columns for summary display
        """
        return self.df[self._mini_summary_columns]

    @property
    def local_survey_id(self) -> str:
        """
        Return string label for local survey id.

        Returns
        -------
        str
            Survey ID for the local station
        """
        survey_id = self.df.loc[~self.df.remote].survey.unique()[0]
        if survey_id in ["none"]:
            survey_id = "0"
        return survey_id

    @property
    def local_survey_metadata(self) -> mt_metadata.timeseries.Survey:
        """Return survey metadata for local station."""
        return self.survey_metadata
        # except KeyError:
        #     msg = f"Unexpected key {self.local_survey_id} not found in survey_metadata"
        #     msg += f"{msg} WARNING -- Maybe old MTH5 -- trying to use key '0'"
        #     logger.warning(msg)
        #     return self.survey_metadata["0"]

    def _add_duration_column(self, df, inplace=True) -> None:
        """Adds a column to self.df with times end-start (in seconds)."""

        timedeltas = df.end - df.start
        durations = [x.total_seconds() for x in timedeltas]
        if inplace:
            df["duration"] = durations
            return df
        else:
            new_df = df.copy()
            new_df["duration"] = durations
            return new_df

    def _update_duration_column(self, inplace=True) -> None:
        """Calls add_duration_column (after possible manual manipulation of start/end."""

        if inplace:
            self._df = self._add_duration_column(self._df, inplace)
        else:
            return self._add_duration_column(self._df, inplace)

    def drop_runs_shorter_than(
        self,
        minimum_duration: float,
        units: str = "s",
        inplace: bool = True,
    ) -> pd.DataFrame | None:
        """
        Drop runs from dataframe that are shorter than minimum duration.

        Parameters
        ----------
        minimum_duration : float
            The minimum allowed duration for a run (in units of units)
        units : str, optional
            Time units, by default "s". Currently only seconds are supported
        inplace : bool, optional
            Whether to modify dataframe in place, by default True

        Returns
        -------
        pd.DataFrame | None
            Modified dataframe if inplace=False, None if inplace=True

        Raises
        ------
        NotImplementedError
            If units other than seconds are specified

        Notes
        -----
        This method needs to have duration refreshed beforehand.
        """
        if units != "s":
            msg = "Expected units are seconds : units='s'"
            raise NotImplementedError(msg)

        drop_cond = self.df.duration < minimum_duration
        if inplace:
            self._update_duration_column(inplace)
            self.df.drop(self.df[drop_cond].index, inplace=inplace)
            self.df.reset_index(drop=True, inplace=True)
            return None
        else:
            new_df = self._update_duration_column(inplace)
            new_df = self.df.drop(self.df[drop_cond].index)
            new_df.reset_index(drop=True, inplace=True)
            return new_df

    def select_station_runs(
        self,
        station_runs_dict: dict,
        keep_or_drop: bool,
        inplace: bool = True,
    ) -> pd.DataFrame | None:
        """
        Partition dataframe based on station_runs_dict and return one partition.

        Parameters
        ----------
        station_runs_dict : dict
            Keys are string IDs of stations to keep/drop.
            Values are lists of string labels for run_ids to keep/drop.
            Example: {"mt01": ["0001", "0003"]}
        keep_or_drop : bool
            If True: returns df with only the station-runs specified
            If False: returns df with station_runs_dict entries removed
        inplace : bool, optional
            If True, modifies dataframe in place, by default True

        Returns
        -------
        pd.DataFrame | None
            Modified dataframe if inplace=False, None if inplace=True
        """

        for station_id, run_ids in station_runs_dict.items():
            if isinstance(run_ids, str):
                run_ids = [
                    run_ids,
                ]
            cond1 = self.df["station"] == station_id
            cond2 = self.df["run"].isin(run_ids)
            if keep_or_drop == "keep":
                drop_df = self.df[cond1 & ~cond2]
            else:
                drop_df = self.df[cond1 & cond2]

        if inplace:
            self.df.drop(drop_df.index, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        else:
            df = self.df.drop(drop_df.index, inplace=False)
            df = df.reset_index(drop=True, inplace=True)
            return df

    def set_run_times(self, run_time_dict: dict, inplace: bool = True):
        """
        Set run times from a dictionary.

        Parameters
        ----------
        run_time_dict : dict
            Dictionary formatted as {run_id: {start, end}}
        inplace : bool, optional
            Whether to modify dataframe in place, by default True

        Returns
        -------
        pd.DataFrame | None
            Modified dataframe if inplace=False, None if inplace=True
        """
        msg = "Need to set run time with a dictionary in the form of {run_id: {start, end}}"
        if not isinstance(run_time_dict, dict):
            raise TypeError(msg)

        for key, times in run_time_dict.items():
            if not isinstance(times, dict):
                raise TypeError(msg)
            if not "start" in times.keys() or "end" not in times.keys():
                raise KeyError(msg)

            cond1 = self.df.run == key
            cond2 = self.df.start <= times["start"]
            cond3 = self.df.end >= times["end"]
            self.df.loc[cond1 & cond2 & cond3, "start"] = times["start"]
            self.df.loc[cond1 & cond2 & cond3, "end"] = times["end"]
        self._update_duration_column()
        self.df = self.restrict_run_intervals_to_simultaneous(self.df)

    @property
    def is_single_station(self) -> bool:
        """Returns True if no RR station."""
        if self.local_station_id:
            if self.remote_station_id:
                return False
            else:
                return True
        else:
            return False

    def restrict_run_intervals_to_simultaneous(self, df: pd.DataFrame) -> None:
        """For each run in local_station_id check if it has overlap with other runs

        There is room for optimization here

        Note that you can wind up splitting runs here.  For example, in that case where
        local is running continuously, but remote is intermittent.  Then the local
        run may break into several chunks.
        :rtype: None
        """
        local_df = df[df.station == self.local_station_id]
        remote_df = df[df.station == self.remote_station_id]
        output_sub_runs = []
        for i_local, local_row in local_df.iterrows():
            for i_remote, remote_row in remote_df.iterrows():
                if intervals_overlap(
                    local_row.start,
                    local_row.end,
                    remote_row.start,
                    remote_row.end,
                ):
                    # print(f"OVERLAP {i_local}, {i_remote}")
                    olap_start, olap_end = overlap(
                        local_row.start,
                        local_row.end,
                        remote_row.start,
                        remote_row.end,
                    )

                    local_sub_run = local_row.copy(deep=True)
                    remote_sub_run = remote_row.copy(deep=True)
                    local_sub_run.start = olap_start
                    local_sub_run.end = olap_end
                    remote_sub_run.start = olap_start
                    remote_sub_run.end = olap_end
                    output_sub_runs.append(local_sub_run)
                    output_sub_runs.append(remote_sub_run)
                else:
                    pass
                    # print(f"NOVERLAP {i_local}, {i_remote}")
        new_df = pd.DataFrame(output_sub_runs)
        new_df = new_df.reset_index(drop=True)

        if new_df.empty:
            msg = (
                f"Local: {self.local_station_id} and "
                f"remote: {self.remote_station_id} do "
                f"not overlap, Remote reference processing not a valid option."
            )
            logger.error(msg)
            raise ValueError(msg)

        return new_df

    def get_station_metadata(
        self, local_station_id: str
    ) -> mt_metadata.timeseries.Station:
        """Returns the station metadata.

        Development Notes:
        TODO: This appears to be unused.  Was probably a precursor to the
          update_survey_metadata() method. Delete if unused. If used fill out doc:
        "Helper function for archiving the TF -- returns an object we can use to populate
        station metadata in the _____"
        :param local_station_id: The name of the local station.
        :type local_station_id: str
        :rtype: mt_metadata.timeseries.Station
        """
        # get a list of local runs:
        cond = self.df["station"] == local_station_id
        sub_df = self.df[cond]
        sub_df.drop_duplicates(subset="run", inplace=True)

        # sanity check:
        run_ids = sub_df.run.unique()
        assert len(run_ids) == len(sub_df)

        station_metadata = sub_df.mth5_obj[0].from_reference(
            sub_df.station_hdf5_reference[0]
        )
        station_metadata.runs = ListDict()
        for i, row in sub_df.iterrows():
            local_run_obj = self.get_run_object(row)
            station_metadata.add_run(local_run_obj.metadata)
        return station_metadata

    def get_run_object(
        self, index_or_row: int | pd.Series
    ) -> mt_metadata.timeseries.Run:
        """
        Get the run object associated with a row of the dataframe.

        Parameters
        ----------
        index_or_row : int | pd.Series
            Row index or row Series from the dataframe

        Returns
        -------
        mt_metadata.timeseries.Run
            The run object associated with the row

        Notes
        -----
        This method may be deprecated in favor of direct calls to
        run_obj = row.mth5_obj.from_reference(row.run_hdf5_reference) in pipelines.
        """
        if isinstance(index_or_row, int):
            row = self.df.loc[index_or_row]
        else:
            row = index_or_row
        run_obj = row.mth5_obj.from_reference(row.run_hdf5_reference)
        return run_obj

    @property
    def num_sample_rates(self) -> int:
        """Returns the number of unique sample rates in the dataframe."""
        return len(self.df.sample_rate.unique())

    @property
    def sample_rate(self) -> float:
        r"""Returns the sample rate that of the data in the dataframe."""
        if self.num_sample_rates != 1:
            msg = "Aurora does not yet process data from mixed sample rates"
            logger.error(f"{msg}")
            raise NotImplementedError(msg)
        sample_rate = self.df.sample_rate.unique()[0]
        return sample_rate

    # this should be deprecated in the future in favor of usin get_metadata_from_df
    def update_survey_metadata(
        self, i: int, row: pd.Series, run_ts: mt_timeseries.run_ts.RunTS
    ) -> None:
        """Wrangle survey_metadata into kernel_dataset.

        Development Notes:
        - The survey metadata needs to be passed to TF before exporting data.
        - This was factored out of initialize_dataframe_for_processing
        - TODO: It looks like we don't need to pass the whole run_ts, just its metadata
           There may be some performance implications to passing the whole object.
           Consider passing run_ts.survey_metadata, run_ts.run_metadata,
           run_ts.station_metadata only
        :param i: This would be the index of row, if we were sure that the dataframe was cleanly indexed.
        :type i: int
        :param row:
        :type row: pd.Series
        :param run_ts: Mth5 object having the survey_metadata.
        :type run_ts: mt_timeseries.run_ts.RunTS
        :rtype: None
        """
        survey_id = run_ts.survey_metadata.id
        if survey_id not in self.survey_metadata.keys():
            self.survey_metadata[survey_id] = run_ts.survey_metadata
        else:
            if row.station in self.survey_metadata[survey_id].stations.keys():
                self.survey_metadata[survey_id].stations[row.station].add_run(
                    run_ts.run_metadata
                )
            else:
                self.survey_metadata[survey_id].add_station(run_ts.station_metadata)
        if len(self.survey_metadata.keys()) > 1:
            raise NotImplementedError

    @property
    def mth5_objs(self):
        """Mth5 objs.
        :return: Dictionary [station_id: mth5_obj].
        :rtype: dict
        """
        mth5_obj_dict = {}
        mth5_obj_dict[self.local_station_id] = self.local_mth5_obj
        if self.remote_station_id is not None:
            mth5_obj_dict[self.remote_station_id] = self.remote_mth5_obj
        return mth5_obj_dict

    def initialize_mth5s(self, mode: str = "r"):
        """
        Return a dictionary of open mth5 objects, keyed by station_id.

        Parameters
        ----------
        mode : str, optional
            File opening mode, by default "r" (read-only)

        Returns
        -------
        dict
            Dictionary keyed by station IDs containing MTH5 objects:
            - local station id: mth5.mth5.MTH5
            - remote station id: mth5.mth5.MTH5 (if present)

        Notes
        -----
        Future versions for multiple station processing may need
        nested dict structure with [survey_id][station].
        """
        self.local_mth5_obj = initialize_mth5(self.local_mth5_path, mode=mode)
        if self.remote_station_id:
            self.remote_mth5_obj = initialize_mth5(self.remote_mth5_path, mode="r")

        self.initialized = True

        return self.mth5_objs

    def initialize_dataframe_for_processing(self) -> None:
        """Adds extra columns needed for processing to the dataframe.

        Populates them with mth5 objects, run_hdf5_reference, and xr.Datasets.

        Development Notes:
        Note #1: When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
        so we convert to xr.DataArray before packing df

        Note #2: [OPTIMIZATION] By accessing the run_ts and packing the "run_dataarray" column of the df, we
         perform a non-lazy operation, and essentially forcing the entire decimation_level=0 dataset to be
         loaded into memory.  Seeking a lazy method to handle this maybe worthwhile.  For example, using
         a df.apply() approach to initialize only one row at a time would allow us to generate the FCs one
         row at a time and never ingest more than one run of data at a time ...

        Note #3: Uncommenting the continue statement here is desireable, will speed things up, but
         is not yet tested.  A nice test would be to have two stations, some runs having FCs built
         and others not having FCs built.  What goes wrong is in update_survey_metadata.
         Need a way to get the survey metadata from a run, not a run_ts if possible
        """

        self.add_columns_for_processing()

        for index, row in self.df.iterrows():
            run_obj = row.mth5_obj.get_run(row.station, row.run, survey=row.survey)
            self.df.loc[index, "run_hdf5_reference"] = run_obj.hdf5_group.ref

            if pd.notna(row.fc) and row.fc:
                msg = f"row {row} already has fcs prescribed by processing config"
                msg += "-- skipping time series initialisation"
                logger.info(msg)
                # see Note #3
                # continue
            # the line below is not lazy, See Note #2
            run_ts = run_obj.to_runts(start=row.start, end=row.end)
            self.df.at[index, "run_dataarray"] = run_ts.dataset.to_array("channel")

            # self.update_survey_metadata(i, row, run_ts)

        logger.info("Dataset dataframe initialized successfully, updated metadata.")

    def add_columns_for_processing(self) -> None:
        """Add columns to the dataframe used during processing.

        Development Notes:
        - This was originally in pipelines.
        - Q: Should mth5_objs be keyed by survey-station?
        - A: Yes, and ...
        since the KernelDataset dataframe will be iterated over, should probably
        write an iterator method.  This can iterate over survey-station tuples
        for multiple station processing.
        - Currently the model of keeping all these data objects "live" in the df
        seems to work OK, but is not well suited to HPC or lazy processing.
        :param mth5_objs: Keys are station_id, values are MTH5 objects.
        :type mth5_objs: dict,
        """
        if not self.initialized:
            raise ValueError("mth5 objects have not been initialized yet.")

        if self._has_df():
            self._df.loc[
                self._df.station == self.local_station_id, "mth5_obj"
            ] = self.local_mth5_obj
            if self.remote_station_id is not None:
                self._df.loc[
                    self._df.station == self.remote_station_id, "mth5_obj"
                ] = self.remote_mth5_obj

    def close_mth5s(self) -> None:
        """Loop over all unique mth5_objs in dataset df and make sure they are closed.+."""
        mth5_objs = self.df["mth5_obj"].unique()
        for mth5_obj in mth5_objs:
            mth5_obj.close_mth5()
        return


def restrict_to_station_list(
    df: pd.DataFrame,
    station_ids: str | list[str],
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Drop all rows where station_ids are NOT in the provided list.

    Operates on a deepcopy of dataframe if inplace=False.

    Parameters
    ----------
    df : pd.DataFrame
        A run summary dataframe
    station_ids : str | list[str]
        Station ids to keep, normally local and remote
    inplace : bool, optional
        If True, modifies dataframe in place, by default True

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only specified stations
    """
    if isinstance(station_ids, str):
        station_ids = [station_ids]
    if not inplace:
        df = copy.deepcopy(df)
    cond1 = ~df["station"].isin(station_ids)
    df.drop(df[cond1].index, inplace=True)
    df = df.reset_index(drop=True)
    return df


def intervals_overlap(
    start1: pd.Timestamp,
    end1: pd.Timestamp,
    start2: pd.Timestamp,
    end2: pd.Timestamp,
) -> bool:
    """Checks if intervals 1, and 2 overlap.

    Interval 1 is (start1, end1), Interval 2 is (start2, end2),

    Development Notes:
    This may work vectorized out of the box but has not been tested.
    Also, it is intended to work with pd.Timestamp objects, but should work
    for many objects that have an ordering associated.
    This website was used as a reference when writing the method:
    https://stackoverflow.com/questions/3721249/python-date-interval-intersection
    :param start1: Start of interval 1.
    :type start1: pd.Timestamp
    :param end1: End of interval 1.
    :type end1: pd.Timestamp
    :param start2: Start of interval 2.
    :type start2: pd.Timestamp
    :param end2: End of interval 2.
    :type end2: pd.Timestamp
    :return cond: True of the intervals overlap, False if they do now.
    :rtype cond: bool
    """
    cond = (start1 <= start2 <= end1) or (start2 <= start1 <= end2)
    return cond


def overlap(
    t1_start: pd.Timestamp,
    t1_end: pd.Timestamp,
    t2_start: pd.Timestamp,
    t2_end: pd.Timestamp,
) -> tuple:
    """Get the start and end times of the overlap between two intervals.

    Interval 1 is (start1, end1), Interval 2 is (start2, end2),

    Development Notes:
     Possibly some nicer syntax in this discussion:
     https://stackoverflow.com/questions/3721249/python-date-interval-intersection
     - Intended to work with pd.Timestamp objects, but should work for many objects
      that have an ordering associated.
    :param t1_start: The start of interval 1.
    :type t1_start: pd.Timestamp
    :param t1_end: The end of interval 1.
    :type t1_end: pd.Timestamp
    :param t2_start: The start of interval 2.
    :type t2_start: pd.Timestamp
    :param t2_end: The end of interval 2.
    :type t2_end: pd.Timestamp
    :return start, end: Start, end are either same type as input, or they are None,None.
    :rtype start, end: tuple
    """
    if t1_start <= t2_start <= t2_end <= t1_end:
        return t2_start, t2_end
    elif t1_start <= t2_start <= t1_end:
        return t2_start, t1_end
    elif t1_start <= t2_end <= t1_end:
        return t1_start, t2_end
    elif t2_start <= t1_start <= t1_end <= t2_end:
        return t1_start, t1_end
    else:
        return None, None
