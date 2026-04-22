# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:31:31 2021

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
import json
import warnings
from io import StringIO

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import numpy as np


# supress the future warning from pandas about using datetime parser.
warnings.simplefilter(action="ignore", category=FutureWarning)


import pandas as pd
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Auxiliary, Electric, Magnetic, Run, Station
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter

from mth5.timeseries import ChannelTS, RunTS


# =============================================================================
def lemi_date_parser(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> pd.Series:
    """
    Combine the date-time columns that are output by LEMI into a single column.

    Assumes UTC timezone.

    Parameters
    ----------
    year : int
        Year value.
    month : int
        Month value (1-12).
    day : int
        Day of the month (1-31).
    hour : int
        Hour in 24-hour format (0-23).
    minute : int
        Minutes in the hour (0-59).
    second : int
        Seconds in the minute (0-59).

    Returns
    -------
    pd.DatetimeIndex
        Combined date-time as a pandas DatetimeIndex.

    """
    dt_df = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
    )

    for key in ["year", "month", "day", "hour", "minute", "second"]:
        dt_df[key] = dt_df[key].astype(int)

    return pd.to_datetime(dt_df)


def lemi_position_parser(position: float) -> float:
    """
    Parse LEMI location strings into decimal degrees.

    Uses the hemisphere for the sign.

    Notes
    -----
    The format of the location is odd in that it is multiplied by
    100 within the LEMI to provide a single floating point value that
    includes the degrees and decimal degrees --> {degrees}{degrees[mm.ss]}.
    For example 40.50166 would be represented as 4030.1.

    Parameters
    ----------
    position : float
        LEMI position value to parse.

    Returns
    -------
    float
        Decimal degrees position.

    """
    pos = f"{float(position) / 100}".split(".")
    degrees = int(pos[0])
    decimals = float(f"{pos[1][0:2]}.{pos[1][2:]}") / 60

    location = degrees + decimals
    return location


def lemi_hemisphere_parser(hemisphere: str) -> int:
    """
    Convert hemisphere into a value [-1, 1].

    Assumes the prime meridian is 0.

    Parameters
    ----------
    hemisphere : str
        Hemisphere string. Valid values are 'N', 'S', 'E', 'W'.

    Returns
    -------
    int
        Unity with a sign for the given hemisphere.
        Returns -1 for 'S' or 'W', 1 for 'N' or 'E'.

    """
    if hemisphere in ["S", "W"]:
        return -1
    return 1


class LEMI424:
    """
    Read and process LEMI424 magnetotelluric data files.

    This is a placeholder until IRIS finalizes their reader.

    Parameters
    ----------
    fn : str or pathlib.Path, optional
        Full path to LEMI424 file, by default None.
    **kwargs : dict
        Additional keyword arguments for configuration.

    Attributes
    ----------
    sample_rate : float
        Sample rate of the file, default is 1.0.
    chunk_size : int
        Chunk size for pandas to use, default is 8640.
    file_column_names : list of str
        Column names of the LEMI424 file.
    dtypes : dict
        Data types for each column.
    data_column_names : list of str
        Same as file_column_names with an added column for date.
    data : pd.DataFrame or None
        The loaded data.

    Notes
    -----
    LEMI424 File Column Names:
        year, month, day, hour, minute, second, bx, by, bz,
        temperature_e, temperature_h, e1, e2, e3, e4, battery,
        elevation, latitude, lat_hemisphere, longitude,
        lon_hemisphere, n_satellites, gps_fix, time_diff

    Data Column Names:
        date, bx, by, bz, temperature_e, temperature_h, e1, e2,
        e3, e4, battery, elevation, latitude, lat_hemisphere,
        longitude, lon_hemisphere, n_satellites, gps_fix, time_diff

    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        self.logger = logger
        self.fn = fn
        self.sample_rate = 1.0
        self.chunk_size = 8640
        self.data = None
        self.file_column_names = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "bx",
            "by",
            "bz",
            "temperature_e",
            "temperature_h",
            "e1",
            "e2",
            "e3",
            "e4",
            "battery",
            "elevation",
            "latitude",
            "lat_hemisphere",
            "longitude",
            "lon_hemisphere",
            "n_satellites",
            "gps_fix",
            "time_diff",
        ]

        self.dtypes = dict(
            [
                ("year", int),
                ("month", int),
                ("day", int),
                ("hour", int),
                ("minute", int),
                ("second", int),
                ("bx", float),
                ("by", float),
                ("bz", float),
                ("temperature_e", float),
                ("temperature_h", float),
                ("e1", float),
                ("e2", float),
                ("e3", float),
                ("e4", float),
                ("battery", float),
                ("elevation", float),
                ("n_satellites", int),
                ("gps_fix", int),
                ("time_diff", float),
            ]
        )

        self.data_column_names = ["date"] + self.file_column_names[6:]

    def __str__(self) -> str:
        """Return string representation of LEMI424 object."""
        lines = ["LEMI 424 data", "-" * 20]
        lines.append(f"start:      {self.start.isoformat()}")
        lines.append(f"end:        {self.end.isoformat()}")
        lines.append(f"N samples:  {self.n_samples}")
        lines.append(f"latitude:   {self.latitude} (degrees)")
        lines.append(f"longitude:  {self.longitude} (degrees)")
        lines.append(f"elevation:  {self.elevation} m")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation of LEMI424 object."""
        return self.__str__()

    def __add__(self, other: "LEMI424 | pd.DataFrame") -> "LEMI424":
        """
        Append other LEMI424 objects or DataFrames together.

        The start and end times should match up for proper concatenation.

        Parameters
        ----------
        other : LEMI424 or pd.DataFrame
            Object to append to this LEMI424 instance.

        Returns
        -------
        LEMI424
            New LEMI424 object with combined data.

        Raises
        ------
        ValueError
            If data is None or if DataFrame columns don't match.

        """
        if not self._has_data():
            raise ValueError("Data is None cannot append to. Read file first")
        if isinstance(other, LEMI424):
            new = LEMI424()
            new.__dict__.update(self.__dict__)
            new.data = pd.concat([new.data, other.data])
            return new
        elif isinstance(other, pd.DataFrame):
            if not other.columns.equals(self.data.columns):
                raise ValueError("DataFrame columns are not the same.")
            new = LEMI424()
            new.__dict__.update(self.__dict__)
            new.data = pd.concat([new.data, other])
            return new
        else:
            raise ValueError(f"Cannot add {type(other)} to pd.DataFrame.")

    @property
    def data(self) -> pd.DataFrame | None:
        """
        Data represented as a pandas DataFrame with data column names.

        Returns
        -------
        pd.DataFrame or None
            The loaded data or None if no data is loaded.

        """
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame | None) -> None:
        """
        Set data ensuring it is a pandas DataFrame with proper column names.

        Parameters
        ----------
        data : pd.DataFrame or None
            Data to set. Must have proper column names if not None.

        Raises
        ------
        ValueError
            If column names don't match expected format.

        """
        if data is None:
            self._data = None
        elif isinstance(data, pd.DataFrame):
            if len(data.columns) == len(self.file_column_names):
                if not data.columns.equals(pd.Index(self.file_column_names)):
                    raise ValueError(
                        "Column names are not the same. "
                        "Check LEMI424.column_names for accepted names"
                    )
            elif len(data.columns) == len(self.data_column_names):
                if not data.columns.equals(pd.Index(self.data_column_names)):
                    raise ValueError(
                        "Column names are not the same. "
                        "Check LEMI424.column_names for accepted names"
                    )
            self._data = data

    def _has_data(self) -> bool:
        """
        Check if object has data loaded.

        Returns
        -------
        bool
            True if data is loaded, False otherwise.

        """
        if self.data is not None:
            return True
        return False

    @property
    def fn(self) -> Path | None:
        """
        Full path to LEMI424 file.

        Returns
        -------
        pathlib.Path or None
            Path to the file or None if not set.

        """
        return self._fn

    @fn.setter
    def fn(self, value: str | Path | None) -> None:
        """
        Set the file path.

        Parameters
        ----------
        value : str, pathlib.Path, or None
            Path to the LEMI424 file.

        Raises
        ------
        IOError
            If the file does not exist.

        """
        if value is not None:
            value = Path(value)
            if not value.exists():
                raise IOError(f"Could not find {value}")
        self._fn = value

    @property
    def file_size(self) -> int | None:
        """
        Size of file in bytes.

        Returns
        -------
        int or None
            File size in bytes or None if no file is set.

        """
        if self.fn is not None:
            return self.fn.stat().st_size

    @property
    def start(self) -> MTime | None:
        """
        Start time of data collection in the LEMI424 file.

        Returns
        -------
        MTime or None
            Start time or None if no data is loaded.

        """
        if self._has_data():
            return MTime(time_stamp=self.data.index[0])

    @property
    def end(self) -> MTime | None:
        """
        End time of data collection in the LEMI424 file.

        Returns
        -------
        MTime or None
            End time or None if no data is loaded.

        """
        if self._has_data():
            return MTime(time_stamp=self.data.index[-1])

    @property
    def latitude(self) -> float | None:
        """
        Median latitude where data have been collected.

        Returns
        -------
        float or None
            Median latitude in degrees or None if no data is loaded.

        """
        if self._has_data():
            return self.data.latitude.median() * self.data.lat_hemisphere.median()

    @property
    def longitude(self) -> float | None:
        """
        Median longitude where data have been collected.

        Returns
        -------
        float or None
            Median longitude in degrees or None if no data is loaded.

        """
        if self._has_data():
            return self.data.longitude.median() * self.data.lon_hemisphere.median()

    @property
    def elevation(self) -> float | None:
        """
        Median elevation where data have been collected.

        Returns
        -------
        float or None
            Median elevation in meters or None if no data is loaded.

        """
        if self._has_data():
            return self.data.elevation.median()

    @property
    def n_samples(self) -> int | None:
        """
        Number of samples in the file.

        Returns
        -------
        int or None
            Number of samples or None if no data/file available.

        """
        if self._has_data():
            return self.data.shape[0]
        elif self.fn is not None and self.fn.exists():
            return round(self.fn.stat().st_size / 152.0)

    @property
    def gps_lock(self) -> Any | None:
        """
        GPS lock status array.

        Returns
        -------
        numpy.ndarray or None
            GPS fix values or None if no data is loaded.

        """
        if self._has_data():
            return self.data.gps_fix.values

    @property
    def station_metadata(self) -> Station:
        """
        Station metadata as mt_metadata.timeseries.Station object.

        Returns
        -------
        mt_metadata.timeseries.Station
            Station metadata object.

        """
        s = Station()
        if self._has_data():
            s.location.latitude = self.latitude
            s.location.longitude = self.longitude
            s.location.elevation = self.elevation
            s.time_period.start = self.start
            s.time_period.end = self.end
            s.add_run(self.run_metadata)
        return s

    @property
    def run_metadata(self) -> Run:
        """
        Run metadata as mt_metadata.timeseries.Run object.

        Returns
        -------
        mt_metadata.timeseries.Run
            Run metadata object.

        """
        r = Run()
        r.id = "a"
        r.sample_rate = self.sample_rate
        r.data_logger.model = "LEMI424"
        r.data_logger.manufacturer = "LEMI"
        if self._has_data():
            r.data_logger.power_source.voltage.start = self.data.battery.max()
            r.data_logger.power_source.voltage.end = self.data.battery.min()
            r.time_period.start = self.start
            r.time_period.end = self.end

            for ch_aux in ["temperature_e", "temperature_h"]:
                r.add_channel(Auxiliary(component=ch_aux))
            for ch_e in ["e1", "e2"]:
                r.add_channel(Electric(component=ch_e))
            for ch_h in ["bx", "by", "bz"]:
                r.add_channel(Magnetic(component=ch_h))
        return r

    def read(self, fn: str | Path | None = None, fast: bool = True) -> None:
        """
        Read a LEMI424 file using pandas.

        The `fast` way will read in the first and last line to get the start
        and end time to make a time index. Then it will read in the data
        skipping parsing the date time columns. It will check to make sure
        the expected amount of points are correct. If not then it will read
        in the slower way which uses the date time parser to ensure any
        time gaps are respected.

        Parameters
        ----------
        fn : str, pathlib.Path, or None, optional
            Full path to file. Uses LEMI424.fn if not provided, by default None.
        fast : bool, optional
            Read the fast way (True) or not (False), by default True.

        Raises
        ------
        IOError
            If file cannot be found.

        """
        if fn is not None:
            self.fn = fn
        if not self.fn.exists():
            msg = f"Could not find file {self.fn}"
            self.logger.error(msg)
            raise IOError(msg)
        if fast:
            try:
                self.read_metadata()
                data = pd.read_csv(
                    self.fn,
                    delimiter=r"\s+",
                    names=self.file_column_names,
                    dtype=self.dtypes,
                    usecols=(
                        "bx",
                        "by",
                        "bz",
                        "temperature_e",
                        "temperature_h",
                        "e1",
                        "e2",
                        "e3",
                        "e4",
                        "battery",
                        "elevation",
                        "latitude",
                        "lat_hemisphere",
                        "longitude",
                        "lon_hemisphere",
                        "n_satellites",
                        "gps_fix",
                        "time_diff",
                    ),
                    converters={
                        "latitude": lemi_position_parser,
                        "longitude": lemi_position_parser,
                        "lat_hemisphere": lemi_hemisphere_parser,
                        "lon_hemisphere": lemi_hemisphere_parser,
                    },
                )
                time_index = pd.date_range(
                    start=self.start.iso_no_tz,
                    end=self.end.iso_no_tz,
                    freq="1000000000ns",
                )
                if time_index.size != data.shape[0]:
                    raise ValueError("Missing a time stamp use read with fast=False")
                data.index = time_index
                self.data = data
                return
            except ValueError:
                self.logger.warning(
                    "Data is missing a time stamp, reading in slow mode"
                )
        # read in chunks, this doesnt really speed up much as most of the
        # compute time is used in the date time parsing.
        if self.n_samples > self.chunk_size:
            st = MTime(time_stamp=None).now()
            dfs = []
            for chunk in pd.read_csv(
                self.fn,
                delimiter=r"\s+",
                names=self.file_column_names,
                dtype=self.dtypes,
                converters={
                    "latitude": lemi_position_parser,
                    "longitude": lemi_position_parser,
                    "lat_hemisphere": lemi_hemisphere_parser,
                    "lon_hemisphere": lemi_hemisphere_parser,
                },
                chunksize=self.chunk_size,
            ):
                # Create date index for this chunk
                chunk.index = lemi_date_parser(
                    chunk["year"],
                    chunk["month"],
                    chunk["day"],
                    chunk["hour"],
                    chunk["minute"],
                    chunk["second"],
                )
                chunk.index.name = "date"
                chunk = chunk.drop(
                    columns=["year", "month", "day", "hour", "minute", "second"]
                )
                dfs.append(chunk)

            if not dfs:
                raise ValueError("File is empty or contains no valid data")

            self.data = pd.concat(dfs)
            et = MTime(time_stamp=None).now()
            self.logger.debug(f"Reading {self.fn.name} took {et - st:.2f} seconds")
        else:
            st = MTime(time_stamp=None).now()
            self.data = pd.read_csv(
                self.fn,
                delimiter=r"\s+",
                names=self.file_column_names,
                dtype=self.dtypes,
                converters={
                    "latitude": lemi_position_parser,
                    "longitude": lemi_position_parser,
                    "lat_hemisphere": lemi_hemisphere_parser,
                    "lon_hemisphere": lemi_hemisphere_parser,
                },
            )

            if self.data.empty:
                raise ValueError("File is empty or contains no valid data")

            # Create date index from individual date/time columns
            self.data.index = lemi_date_parser(
                self.data["year"],
                self.data["month"],
                self.data["day"],
                self.data["hour"],
                self.data["minute"],
                self.data["second"],
            )
            self.data.index.name = "date"
            self.data = self.data.drop(
                columns=["year", "month", "day", "hour", "minute", "second"]
            )

            et = MTime(time_stamp=None).now()
            self.logger.debug(f"Reading {self.fn.name} took {et - st:.2f} seconds")

    def read_metadata(self) -> None:
        """
        Read only first and last rows to get important metadata.

        This method is used to extract essential metadata from the collection
        without loading the entire dataset.

        """

        with open(self.fn) as fid:
            first_line = fid.readline()
            last_line = first_line  # Default to first line for single-line files
            for line in fid:
                last_line = line  # Update for multi-line files

        if not first_line.strip():
            raise ValueError("File is empty or contains no valid data")

        lines = StringIO(f"{first_line}\n{last_line}")

        self.data = pd.read_csv(
            lines,
            delimiter=r"\s+",
            names=self.file_column_names,
            dtype=self.dtypes,
            converters={
                "latitude": lemi_position_parser,
                "longitude": lemi_position_parser,
                "lat_hemisphere": lemi_hemisphere_parser,
                "lon_hemisphere": lemi_hemisphere_parser,
            },
        )

        # Create date index from individual date/time columns
        self.data.index = lemi_date_parser(
            self.data["year"],
            self.data["month"],
            self.data["day"],
            self.data["hour"],
            self.data["minute"],
            self.data["second"],
        )
        self.data.index.name = "date"
        self.data = self.data.drop(
            columns=["year", "month", "day", "hour", "minute", "second"]
        )

    def read_calibration(self, fn: str | Path) -> FrequencyResponseTableFilter:
        """
        Read a LEMI424 calibration file.

        Calibration files are assumed to be JSON files with the following format:
        {
            "Calibration": {
                "gain": float,
                "Freq": [float],
                "Re": [float],
                "Im": [float]
            }
        }


        Parameters
        ----------
        fn : str or pathlib.Path
            Full path to calibration file.

        Returns
        -------
        mt_metadata.timeseries.filters.FrequencyResponseTableFilter
            Calibration filter object.

        """

        with open(fn, "r") as cf:
            try:
                cal_data = json.load(cf)["Calibration"]

            except json.JSONDecodeError as e:
                self.logger.error(f"Error reading calibration file {fn}: {e}")
                raise
            except KeyError as e:
                self.logger.error(f"Calibration key not found in file {fn}: {e}")
                raise

        gain = cal_data.get("gain", 1.0)
        frequencies = np.array(cal_data.get("Freq", []))
        real = np.array(cal_data.get("Re", []))
        imag = np.array(cal_data.get("Im", []))

        if real.size > 0 and imag.size > 0:
            amplitudes = np.sqrt(real**2 + imag**2)
            phases = np.degrees(np.arctan2(imag, real))

        cal_filter = FrequencyResponseTableFilter(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            gain=gain,
            instrument_type="flux gate magnetometer",
            units_in="nanoTesla",
            units_out="nanoTesla",
            sequence_number=1,
        )
        return cal_filter

    def to_run_ts(
        self,
        fn: str | Path | None = None,
        e_channels: list[str] = ["e1", "e2"],
        calibration_dict: dict | None = None,
    ) -> RunTS:
        """
        Create a RunTS object from the data.

        Parameters
        ----------
        fn : str, pathlib.Path, or None, optional
            Full path to file. Will use LEMI424.fn if None, by default None.
        e_channels : list of str, optional
            Column names for the electric channels to use, by default ["e1", "e2"].
        calibration_dict : dict, optional
            Calibration dictionary to apply to the data, by default {}.  Keys are
            the channel names and values are the calibration file path. The file
            path is assumed to be in the format `lemi-{component}.sr.json`.

        Returns
        -------
        mth5.timeseries.RunTS
            RunTS object containing the data.

        """

        ch_list = []
        if calibration_dict is None:
            calibration_dict = {}
        if not isinstance(calibration_dict, dict):
            raise ValueError("calibration_dict must be a dictionary")

        for comp in (
            ["bx", "by", "bz"] + e_channels + ["temperature_e", "temperature_h"]
        ):
            channel_response = None
            if comp in calibration_dict.keys():
                fap_filter = self.read_calibration(calibration_dict[comp])
                fap_filter.name = f"lemi424_{comp}_calibration"
                channel_response = ChannelResponse(filters_list=[fap_filter])

            if comp[0] in ["h", "b"]:
                ch = ChannelTS("magnetic")
                ch.channel_metadata.units = "nT"
            elif comp[0] in ["e"]:
                ch = ChannelTS("electric")
                ch.channel_metadata.units = "mV/km"
            else:
                ch = ChannelTS("auxiliary")
                ch.channel_metadata.units = "C"
            ch.sample_rate = self.sample_rate
            ch.start = self.start
            ch.ts = self.data[comp].values
            ch.component = comp
            if channel_response is not None:
                ch.channel_response = channel_response

            ch_list.append(ch)
        return RunTS(
            array_list=ch_list,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
        )


# =============================================================================
# define the reader
# =============================================================================
def read_lemi424(
    fn: str | Path | list[str | Path],
    e_channels: list[str] = ["e1", "e2"],
    fast: bool = True,
    calibration_dict: dict | None = None,
) -> RunTS:
    """
    Read a LEMI 424 TXT file.

    Parameters
    ----------
    fn : str or pathlib.Path
        Input file name.
    e_channels : list of str, optional
        A list of electric channels to read, by default ["e1", "e2"].
    fast : bool, optional
        Use fast reading method, by default True.
    calibration_dict : dict, optional
        Calibration dictionary to apply to the data, by default None.  Keys are
        the channel names and values are the calibration file path.

    Returns
    -------
    mth5.timeseries.RunTS
        A RunTS object with appropriate metadata.

    """

    if not isinstance(fn, (list, tuple)):
        fn = [fn]
    txt_obj = LEMI424(fn[0])
    txt_obj.read(fast=fast)

    # read a list of files into a single run
    if len(fn) > 1:
        for txt_file in fn[1:]:
            other = LEMI424(txt_file)
            other.read(fast=fast)
            txt_obj += other
    return txt_obj.to_run_ts(e_channels=e_channels, calibration_dict=calibration_dict)
