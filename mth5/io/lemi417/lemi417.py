# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 2026

:copyright:
    XuBingcheng (xubc1024@foxmail.com)
    Modified from "lemi424.py" (by Jared Peacock)

:license: MIT
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from mt_metadata.timeseries import Station, Run, Electric, Magnetic, Auxiliary
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter
from mt_metadata.utils.mttime import MTime

from mth5.timeseries import ChannelTS, RunTS

# =============================================================================

class LEMI417:
    """
    Read and process LEMI-417 magnetotelluric binary data files.

    File structure involves 512-byte blocks with a 32-byte header (tag)
    followed by a 480-byte data segment (16 samples × 30 bytes).

    Parameters
    ----------
    fn : str or pathlib.Path, optional
        Full path to LEMI417 binary file, by default None.
    conv_factors : numpy.ndarray, optional
        Conversion factors for the 7 channels, by default ones.

    Attributes
    ----------
    header_list : list of dict
        Metadata extracted from each block's tag.
    data : pd.DataFrame or None
        The loaded and parsed data.
    sample_rate : float
        Sample rate calculated from the file headers.
    """

    def __init__(
        self, 
        fn: Optional[Union[str, Path]] = None, 
        conv_factors: Optional[np.ndarray] = None
    ) -> None:
        self.logger = logger
        self._fn = None
        self.header_list: List[Dict[str, Any]] = []
        self.conv_factors = conv_factors if conv_factors is not None else np.ones(7)
        self.data: Optional[pd.DataFrame] = None
        self.sample_rate: Optional[float] = None
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        self.fn = fn

    def __str__(self) -> str:
        """Return string representation of LEMI417 object."""
        lines = ["LEMI 417 data", "-" * 20]
        if self._has_data():
            lines.append(f"start:      {self.start.isoformat()}")
            lines.append(f"end:        {self.end.isoformat()}")
            lines.append(f"N samples:  {self.n_samples}")
            lines.append(f"latitude:   {self.latitude} (degrees)")
            lines.append(f"longitude:  {self.longitude} (degrees)")
            lines.append(f"elevation:  {self.elevation} m")
        else:
            lines.append("No data loaded")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation of LEMI417 object."""
        return self.__str__()

    def __add__(self, other: LEMI417) -> LEMI417:
        """
        Append another LEMI417 object with time continuity check.

        Parameters
        ----------
        other : LEMI417
            The other LEMI417 instance to concatenate.

        Returns
        -------
        LEMI417
            A new LEMI417 instance with combined data.

        Raises
        ------
        TypeError
            If 'other' is not an instance of LEMI417.
        ValueError
            If either object lacks data or if there's a significant time gap.
        """
        if not isinstance(other, LEMI417):
            raise TypeError(f"Cannot add LEMI417 with {type(other)}")

        if self.data is None or other.data is None:
            raise ValueError("Both LEMI417 objects must have data")

        dt = 1.0 / self.sample_rate
        tol = 8.0  # seconds tolerance

        expected = self.end + timedelta(seconds=dt)
        actual = other.start

        if abs((actual - expected).total_seconds()) > tol:
            raise ValueError(
                "Time gap detected between files:\n"
                f"previous end = {self.end}\n"
                f"current start = {other.start}"
            )

        new = LEMI417()
        new.__dict__.update(self.__dict__)
        new.data = pd.concat([self.data, other.data])
        new._start = self.start
        new._end = other.end
        new.header_list = self.header_list + other.header_list

        return new

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def _has_data(self) -> bool:
        """Check if data has been loaded and is not empty."""
        return self.data is not None and not self.data.empty

    @property
    def fn(self) -> Optional[Path]:
        """Full path to LEMI417 binary file."""
        return self._fn

    @fn.setter
    def fn(self, value: Optional[Union[str, Path]]) -> None:
        """Set file path and validate existence."""
        if value is not None:
            value = Path(value)
            if not value.exists():
                raise IOError(f"File not found: {value}")
        self._fn = value

    @property
    def file_size(self) -> Optional[int]:
        """Size of file in bytes."""
        if self.fn is not None:
            return self.fn.stat().st_size
        return None

    @property
    def start(self) -> Optional[datetime]:
        """Start time of data collection."""
        return self._start

    @property
    def end(self) -> Optional[datetime]:
        """End time of data collection."""
        return self._end

    @property
    def n_samples(self) -> int:
        """Number of samples in the loaded data."""
        return 0 if self.data is None else len(self.data)

    @property
    def header_df(self) -> pd.DataFrame:
        """Metadata headers as a pandas DataFrame."""
        if not self.header_list:
            return pd.DataFrame()
        return pd.DataFrame(self.header_list)

    @property
    def latitude(self) -> Optional[float]:
        """Median latitude collected during the survey."""
        if self._has_data() and not self.header_df.empty:
            return self.header_df["latitude"].median()
        return None

    @property
    def longitude(self) -> Optional[float]:
        """Median longitude collected during the survey."""
        if self._has_data() and not self.header_df.empty:
            return self.header_df["longitude"].median()
        return None

    @property
    def elevation(self) -> Optional[float]:
        """Median elevation collected during the survey."""
        if self._has_data() and not self.header_df.empty:
            return self.header_df["elevation"].median()
        return None

    @property
    def station_metadata(self) -> Station:
        """Station metadata as an mt_metadata.timeseries.Station object."""
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
        """Run metadata as an mt_metadata.timeseries.Run object."""
        r = Run()
        r.id = "a"
        r.sample_rate = self.sample_rate
        r.data_logger.model = "LEMI417"
        r.data_logger.manufacturer = "LEMI"
        if self._has_data() and not self.header_df.empty:
            r.data_logger.power_source.voltage.start = float(self.header_df["vin"].max())
            r.data_logger.power_source.voltage.end = float(self.header_df["vin"].min())
            r.time_period.start = self.start
            r.time_period.end = self.end

            for ch_aux in ["temperature_h", "temperature_e"]:
                r.add_channel(Auxiliary(component=ch_aux))
            for ch_e in ["e1", "e2"]:
                r.add_channel(Electric(component=ch_e))
            for ch_h in ["bx", "by", "bz"]:
                r.add_channel(Magnetic(component=ch_h))
        return r

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    @staticmethod
    def _bcd_to_int(x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert BCD (Binary Coded Decimal) to integer."""
        return (x >> 4) * 10 + (x & 0x0F)

    @staticmethod
    def int2float24(x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert 24-bit integer to float, handling signedness."""
        x = np.asarray(x, dtype=np.float64)
        threshold_23 = 2**23 - 1
        full_24 = 2**24
        x[x > threshold_23] -= full_24
        x /= 100.0
        return x.item() if x.size == 1 else x

    @staticmethod
    def int2float32(x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert 32-bit integer to float, handling signedness."""
        x = np.asarray(x, dtype=np.float64)
        threshold_31 = 2**31 - 1
        full_32 = 2**32
        x[x > threshold_31] -= full_32
        x /= 100.0
        return x.item() if x.size == 1 else x

    @staticmethod
    def int2float16(x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert 16-bit integer to float, handling signedness."""
        x = np.asarray(x, dtype=np.float64)
        threshold_15 = 2**15 - 1
        full_16 = 2**16
        x[x > threshold_15] -= full_16
        x /= 100.0
        return x.item() if x.size == 1 else x

    @staticmethod
    def latitude_position(position: np.ndarray) -> float:
        """Parse raw latitude from BCD bytes."""
        degrees = int(position[0])
        decimals = float(f"{position[1]}.{position[2]}{position[3]}") / 60
        return degrees + decimals

    @staticmethod
    def longitude_position(position: np.ndarray) -> float:
        """Parse raw longitude from BCD bytes."""
        degrees = int(position[0]) * 100 + int(position[1])
        decimals = float(f"{position[2]}.{position[3]}{position[4]}") / 60
        return degrees + decimals

    @staticmethod
    def hemisphere_parser(hemisphere: str) -> int:
        """Convert hemisphere string [N, S, E, W] to sign [-1, 1]."""
        if hemisphere in ["S", "W"]:
            return -1
        return 1

    # ------------------------------------------------------------------
    # Core Reader and Calibration
    # ------------------------------------------------------------------
    def read(self, fn: Optional[Union[str, Path]] = None) -> LEMI417:
        """
        Read a LEMI417 binary file.

        The binary format consists of 512-byte blocks. Each block has 32 bytes
        of header and 480 bytes of data (16 samples).

        Parameters
        ----------
        fn : str or pathlib.Path, optional
            Path to the file. Uses self.fn if not provided.

        Returns
        -------
        LEMI417
            Returns self for method chaining.
        """
        if fn is not None:
            self.fn = fn
        if self.fn is None:
            raise ValueError("No input file specified")

        self.header_list = []

        with open(self.fn, "rb") as f:
            f.seek(0, 2)
            fsize = f.tell()
            f.seek(0)

            nblock = fsize // 512
            if nblock == 0:
                raise ValueError(f"File {self.fn} contains no valid data blocks.")
            
            nsample = nblock * 16
            alldata = np.zeros((nsample, 9), dtype=np.float64)
            all_times = np.empty(nsample, dtype=object)

            for iblock in range(nblock):
                # Header Parsing
                tag = np.frombuffer(f.read(32), dtype=np.uint8)
                
                YY = int(self._bcd_to_int(tag[5]))
                MM = int(self._bcd_to_int(tag[6]))
                DD = int(self._bcd_to_int(tag[7]))
                hh = int(self._bcd_to_int(tag[8]))
                mm = int(self._bcd_to_int(tag[9]))
                ss = int(self._bcd_to_int(tag[10]))

                fs_val = tag[25]
                fs_rate = 4.0 / fs_val
                self.sample_rate = fs_rate

                t0 = datetime(2000 + YY, MM, DD, hh, mm, ss)

                # Geographic Data
                lat_raw = self._bcd_to_int(tag[11:15])
                lat_hem = chr(tag[15]).upper()
                lat = self.latitude_position(lat_raw) * self.hemisphere_parser(lat_hem)

                lon_raw = self._bcd_to_int(tag[16:21])
                lon_hem = chr(tag[21]).upper()
                lon = self.longitude_position(lon_raw) * self.hemisphere_parser(lon_hem)

                elev = int(self._bcd_to_int(tag[23])) * 1000 + int(self._bcd_to_int(tag[24])) * 10

                current_header = {
                    "block_index": iblock,
                    "model": bytes(tag[0:4]).decode("ascii", errors="ignore"),
                    "serial_number": int(self._bcd_to_int(tag[4])),
                    "block_start_time": t0,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                    "vin": tag[26] / 10.0,
                    "vbat": tag[27] / 10.0,
                    "electrode_lengths": tag[28:32].tolist(),
                    "fs": fs_val,
                    "sample_rate": fs_rate
                }
                self.header_list.append(current_header)

                # Data Segment Parsing
                raw_data = f.read(480)
                if len(raw_data) < 480:
                    self.logger.warning(f"Truncated block at index {iblock}")
                    break
                
                raw = np.frombuffer(raw_data, dtype=np.uint8).reshape(16, 30).astype(int)
                data_block = np.zeros((16, 9), dtype=np.float64)

                # Magnetic (Bx, By, Bz) - 24 bit
                for ch in range(3):
                    data_block[:, ch] = self.int2float24(
                        raw[:, ch * 3] + raw[:, ch * 3 + 1] * 256 + raw[:, ch * 3 + 2] * 256**2
                    )

                # Electric (E1, E2, E3, E4) - 32 bit
                for ch in range(4):
                    off = 9 + ch * 4
                    data_block[:, 3 + ch] = self.int2float32(
                        raw[:, off] + raw[:, off + 1] * 256 + raw[:, off + 2] * 256**2 + raw[:, off + 3] * 256**3
                    )

                # Auxiliary (Temperature H, Temperature E) - 16 bit
                for ch in range(2):
                    off = 25 + ch * 2
                    data_block[:, 7 + ch] = self.int2float16(
                        raw[:, off] + raw[:, off + 1] * 256
                    )

                start_idx = iblock * 16
                end_idx = start_idx + 16
                alldata[start_idx:end_idx, :] = data_block

                time_offsets = np.arange(16) / fs_rate
                block_times = [t0 + timedelta(seconds=off) for off in time_offsets]
                all_times[start_idx:end_idx] = block_times

        if self.header_list:
            # Construct DataFrame
            self.data = pd.DataFrame(
                alldata[:len(self.header_list)*16],
                index=pd.DatetimeIndex(all_times[:len(self.header_list)*16]),
                columns=["bx", "by", "bz", "e1", "e2", "e3", "e4", "temperature_h", "temperature_e"],
            )
            self._start = self.data.index[0]
            self._end = self.data.index[-1]
        else:
            self.data = pd.DataFrame()
            self._start = self._end = None

        return self

    def read_calibration(self, fn: Union[str, Path]) -> FrequencyResponseTableFilter:
        """
        Read a LEMI417 calibration JSON file.

        Format expected:
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
            Path to calibration file.

        Returns
        -------
        FrequencyResponseTableFilter
            The parsed calibration filter.
        """
        fn = Path(fn)
        with open(fn, "r") as f:
            cal = json.load(f)["Calibration"]

        gain = cal.get("gain", 1.0)
        freqs = np.array(cal.get("Freq", []))
        real = np.array(cal.get("Re", []))
        imag = np.array(cal.get("Im", []))

        amps = np.sqrt(real**2 + imag**2)
        phases = np.degrees(np.arctan2(imag, real))

        return FrequencyResponseTableFilter(
            frequencies=freqs,
            amplitudes=amps,
            phases=phases,
            gain=gain,
            instrument_type="lemi417",
            units_in="nanoTesla",
            units_out="nanoTesla",
        )

    def to_run_ts(
        self, 
        e_channels: List[str] = ["e1", "e2"],
        calibration_dict: Optional[Dict[str, Union[str, Path]]] = None
    ) -> RunTS:
        """
        Convert LEMI417 data to an MTH5 RunTS object.

        Parameters
        ----------
        e_channels : list of str, optional
            Electric channels to include, by default ["e1", "e2"].
        calibration_dict : dict, optional
            Mapping of channel names to calibration file paths.

        Returns
        -------
        RunTS
            MTH5 RunTS object containing the time series and metadata.
        """
        if not self._has_data():
            raise ValueError("No data loaded. Call read() first.")

        if calibration_dict is None:
            calibration_dict = {}

        all_comps = ["bx", "by", "bz"] + e_channels + ["temperature_h", "temperature_e"]
        ch_list = []

        for comp in all_comps:
            if comp not in self.data.columns:
                self.logger.warning(f"Channel {comp} not found in data.")
                continue

            response = None
            if comp in calibration_dict:
                fap = self.read_calibration(calibration_dict[comp])
                fap.name = f"lemi417_{comp}_calibration"
                response = ChannelResponse(filters_list=[fap])

            if comp[0] in ["h", "b"]:
                ch = ChannelTS("magnetic", units="nT")
            elif comp[0] in ["e"]:
                ch = ChannelTS("electric", units="mV/km")
            else:
                ch = ChannelTS("auxiliary", units="C")
            
            ch.sample_rate = self.sample_rate
            ch.start = self.start
            ch.ts = self.data[comp].values
            ch.component = comp
            if response:
                ch.channel_response = response
            
            ch_list.append(ch)

        return RunTS(
            array_list=ch_list,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
        )

# =============================================================================
# Factory Function
# =============================================================================

def read_lemi417(
    fn: Union[str, Path, List[Union[str, Path]]], 
    conv_factors: Optional[np.ndarray] = None, 
    e_channels: List[str] = ["e1", "e2"],
    calibration_dict: Optional[Dict[str, Union[str, Path]]] = None
) -> RunTS:
    """
    Read LEMI 417 binary files and return a RunTS object.

    Parameters
    ----------
    fn : str, Path or list of str/Path
        Input file(s).
    conv_factors : numpy.ndarray, optional
        Conversion factors.
    e_channels : list of str, optional
        Electric channels to read, default is ["e1", "e2"].
    calibration_dict : dict, optional
        Calibration dictionary.

    Returns
    -------
    mth5.timeseries.RunTS
        A RunTS object with appropriate metadata and time series.
    """
    if isinstance(fn, (str, Path)):
        fn = [fn]

    obj = LEMI417(fn[0], conv_factors=conv_factors)
    obj.read()

    for f_path in fn[1:]:
        other = LEMI417(f_path, conv_factors=conv_factors)
        other.read()
        obj = obj + other

    return obj.to_run_ts(e_channels=e_channels, calibration_dict=calibration_dict)
