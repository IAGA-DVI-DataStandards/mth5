# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:03:23 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import inspect
import weakref
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mt_metadata import timeseries as metadata
from mt_metadata.base import MetadataBase
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import ChannelResponse
from mt_timeseries import ChannelTS
from mt_timeseries.channel_ts import make_dt_coordinates

from mth5 import CHANNEL_DTYPE
from mth5.groups import FiltersGroup
from mth5.helpers import (
    add_attributes_to_metadata_class_pydantic,
    from_numpy_type,
    inherit_doc_string,
    read_attrs_to_dict,
    to_numpy_type,
)
from mth5.utils.exceptions import MTH5Error


meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))


# =============================================================================
class ChannelDataset:
    """
    A container for channel time series data stored in HDF5 format.

    This class provides a flexible interface to work with magnetotelluric channel data,
    allowing conversion to various formats (xarray, pandas, numpy) while maintaining
    metadata integrity.

    Parameters
    ----------
    dataset : h5py.Dataset or None
        HDF5 dataset object containing the channel time series data.
    dataset_metadata : MetadataBase, optional
        Metadata container for Electric, Magnetic, or Auxiliary channel types.
        Default is None.
    write_metadata : bool, optional
        Whether to write metadata to the HDF5 dataset on initialization.
        Default is True.
    **kwargs : dict
        Additional keyword arguments to set as instance attributes.

    Attributes
    ----------
    hdf5_dataset : h5py.Dataset
        Weak reference to the underlying HDF5 dataset.
    metadata : MetadataBase
        Channel metadata object with validation.
    logger : loguru.Logger
        Logger instance for tracking operations.

    Raises
    ------
    MTH5Error
        If the dataset is not of the correct type or metadata validation fails.

    See Also
    --------
    ElectricDataset : Specialized container for electric field channels.
    MagneticDataset : Specialized container for magnetic field channels.
    AuxiliaryDataset : Specialized container for auxiliary channels.

    Examples
    --------
    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
    >>> channel = run.get_channel('Ex')
    >>> channel
    Channel Electric:
    -------------------
      component:        Ex
      data type:        electric
      data format:      float32
      data shape:       (4096,)
      start:            1980-01-01T00:00:00+00:00
      end:              1980-01-01T00:00:01+00:00
      sample rate:      4096

    Access time series data

    >>> ts_data = channel.to_channel_ts()
    >>> print(f"Mean: {ts_data.ts.mean():.2f}, Std: {ts_data.ts.std():.2f}")

    Convert to xarray for time-based indexing

    >>> xr_data = channel.to_xarray()
    >>> subset = xr_data.sel(time=slice('1980-01-01T00:00:00', '1980-01-01T00:00:10'))

    """

    def __init__(
        self,
        dataset: h5py.Dataset | None,
        dataset_metadata: MetadataBase | None = None,
        write_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        try:
            metadata_obj = meta_classes[self._class_name]
        except KeyError:
            metadata_obj = MetadataBase
        # add mth5 attributes to the metadata class
        self.metadata = add_attributes_to_metadata_class_pydantic(metadata_obj)
        self.metadata.hdf5_reference = self.hdf5_dataset.ref
        self.metadata.mth5_type = self._class_name
        # if the input data set already has filled attributes, namely if the
        # channel data already exists then read them in without writing back
        if "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.read_metadata()

            # this causes issues because the attrs are in binary format
            # self.metadata.from_dict(
            #     {self.hdf5_dataset.attrs["mth5_type"]: dict(self.hdf5_dataset.attrs)}
            # )
        # if metadata is input, make sure that its the same class type amd write
        # to the hdf5 dataset
        if dataset_metadata is not None:
            if not isinstance(self.metadata, type(dataset_metadata)):
                msg = (
                    f"metadata must be type metadata.{self._class_name} not "
                    f"{type(dataset_metadata)}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            # load from dict because of the extra attributes for MTH5
            # Filter out None values for mth5_type to avoid pydantic validation errors
            metadata_dict = dataset_metadata.to_dict()
            # Clean the dict to remove None values for mth5_type that might cause validation errors
            for class_key, class_data in metadata_dict.items():
                if isinstance(class_data, dict) and class_data.get("mth5_type") is None:
                    class_data.pop("mth5_type", None)

            self.metadata.from_dict(metadata_dict)
            # Always set these critical properties regardless of the path
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            if write_metadata:
                self.write_metadata()
        else:
            self.read_metadata()
        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.hdf5_dataset.attrs["mth5_type"] = self._class_name
            self.write_metadata()

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the channel.

        Returns
        -------
        str
            Formatted string with channel metadata and data information.

        Examples
        --------
        >>> print(channel)
        Channel Electric:
        -------------------
          component:        Ex
          data type:        electric
          data format:      float32
          data shape:       (4096,)
          start:            1980-01-01T00:00:00+00:00
          end:              1980-01-01T00:00:01+00:00
          sample rate:      4096
        """
        try:
            lines = ["Channel {0}:".format(self._class_name)]
            lines.append("-" * (len(lines[0]) + 2))
            info_str = "\t{0:<18}{1}"
            lines.append(info_str.format("component:", self.metadata.component))
            lines.append(info_str.format("data type:", self.metadata.type))
            lines.append(info_str.format("data format:", self.hdf5_dataset.dtype))
            lines.append(info_str.format("data shape:", self.hdf5_dataset.shape))
            lines.append(info_str.format("start:", self.metadata.time_period.start))
            lines.append(info_str.format("end:", self.metadata.time_period.end))
            lines.append(info_str.format("sample rate:", self.metadata.sample_rate))
            return "\n".join(lines)
        except ValueError:
            return "MTH5 file is closed and cannot be accessed."

    def __repr__(self) -> str:
        """
        Return the string representation of the channel.

        Returns
        -------
        str
            String representation identical to __str__.
        """
        return self.__str__()

    @property
    def _class_name(self) -> str:
        """
        Extract the base class name without 'Dataset' suffix.

        Returns
        -------
        str
            Base class name (e.g., 'Electric', 'Magnetic', 'Auxiliary').
        """
        return self.__class__.__name__.split("Dataset")[0]

    @property
    def run_metadata(self) -> metadata.Run:
        """
        Get the run-level metadata containing this channel.

        Returns
        -------
        metadata.Run
            Run metadata object with channel information included.

        Examples
        --------
        >>> run_meta = channel.run_metadata
        >>> print(run_meta.id)
        'MT001a'
        >>> print(run_meta.channels_recorded_electric)
        ['Ex', 'Ey']
        """
        meta_dict = dict(self.hdf5_dataset.parent.attrs)
        for key, value in meta_dict.items():
            meta_dict[key] = from_numpy_type(value)
        run_metadata = metadata.Run()
        run_metadata.from_dict({"run": meta_dict})
        run_metadata.add_channel(self.metadata)
        return run_metadata

    @property
    def station_metadata(self) -> metadata.Station:
        """
        Get the station-level metadata containing this channel.

        Returns
        -------
        metadata.Station
            Station metadata object with run and channel information.

        Examples
        --------
        >>> station_meta = channel.station_metadata
        >>> print(f"{station_meta.id}: {station_meta.location.latitude}, {station_meta.location.longitude}")
        'MT001: 40.5, -112.3'
        """
        station_metadata = metadata.Station()
        station_metadata.from_dict(
            {
                "station": read_attrs_to_dict(
                    dict(self.hdf5_dataset.parent.parent.attrs), station_metadata
                )
            }
        )
        station_metadata.add_run(self.run_metadata)
        return station_metadata

    @property
    def survey_metadata(self) -> metadata.Survey:
        """
        Get the survey-level metadata containing this channel.

        Returns
        -------
        metadata.Survey
            Complete survey metadata hierarchy including this channel.

        Examples
        --------
        >>> survey_meta = channel.survey_metadata
        >>> print(survey_meta.id)
        'MT Survey 2023'
        >>> print(f"Stations: {len(survey_meta.stations)}")
        Stations: 15
        """
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict(
            {
                "survey": read_attrs_to_dict(
                    dict(self.hdf5_dataset.parent.parent.parent.parent.attrs),
                    survey_metadata,
                )
            }
        )
        survey_metadata.add_station(self.station_metadata)
        return survey_metadata

    @property
    def survey_id(self) -> str:
        """
        Get the survey identifier.

        Returns
        -------
        str
            Survey ID string.

        Examples
        --------
        >>> print(channel.survey_id)
        'MT_Survey_2023'
        """
        return self.hdf5_dataset.parent.parent.parent.parent.attrs["id"]

    @property
    def channel_response(self) -> ChannelResponse:
        """
        Get the complete channel response from applied filters.

        Constructs a ChannelResponse object by retrieving all filters referenced
        in the channel metadata from the survey's Filters group.

        Returns
        -------
        ChannelResponse
            Channel response object containing all applied filters in sequence.

        Notes
        -----
        Filters are applied in the order specified by their sequence_number.
        Filter names are normalized by replacing '/' with ' per ' and converting
        to lowercase.

        Examples
        --------
        >>> response = channel.channel_response
        >>> print(f"Number of filters: {len(response.filters_list)}")
        Number of filters: 3
        >>> for filt in response.filters_list:
        ...     print(f"{filt.name}: {filt.type}")
        zpk: zpk
        coefficient: coefficient
        time delay: time_delay
        """
        # get the filters to make a channel response
        filters_group = FiltersGroup(
            self.hdf5_dataset.parent.parent.parent.parent["Filters"]
        )
        f_list = []
        # Check if filters field exists (new version with AppliedFilter objects)
        if hasattr(self.metadata, "filters") and self.metadata.filters:
            for count, applied_filter in enumerate(self.metadata.filters, 1):
                if hasattr(applied_filter, "name"):
                    name = applied_filter.name.replace("/", " per ").lower()
                    try:
                        filt_obj = filters_group.to_filter_object(name)
                        filt_obj.sequence_number = count
                        f_list.append(filt_obj)
                    except KeyError:
                        self.logger.warning(f"Could not locate filter {name}")
                        continue
        # Fallback to old filter field for backward compatibility
        elif hasattr(self.metadata, "filter") and hasattr(self.metadata.filter, "name"):
            for count, name in enumerate(self.metadata.filter.name, 1):
                name = name.replace("/", " per ").lower()
                try:
                    filt_obj = filters_group.to_filter_object(name)
                    filt_obj.sequence_number = count
                    f_list.append(filt_obj)
                except KeyError:
                    self.logger.warning(f"Could not locate filter {name}")
                    continue
        return ChannelResponse(filters_list=f_list)

    @property
    def start(self) -> MTime:
        """
        Get the start time of the channel data.

        Returns
        -------
        MTime
            Start time from metadata.time_period.start.

        Examples
        --------
        >>> print(channel.start)
        1980-01-01T00:00:00+00:00
        >>> print(channel.start.iso_str)
        '1980-01-01T00:00:00.000000+00:00'
        """
        return self.metadata.time_period.start

    @start.setter
    def start(self, value: str | MTime) -> None:
        """
        Set the start time with validation.

        Parameters
        ----------
        value : str or MTime
            New start time in ISO format string or MTime object.

        Examples
        --------
        >>> channel.start = '1980-01-01T12:00:00'
        >>> channel.start = MTime('1980-01-01T12:00:00')
        """
        if isinstance(value, MTime):
            self.metadata.time_period.start = value.isoformat()
        else:
            self.metadata.time_period.start = value

    @property
    def end(self) -> MTime:
        """
        Calculate the end time based on start time, sample rate, and number of samples.

        Returns
        -------
        MTime
            Calculated end time of the data.

        Notes
        -----
        End time is calculated as: start + (n_samples - 1) / sample_rate
        The -1 ensures the last sample falls exactly at the end time.

        Examples
        --------
        >>> print(f"Duration: {channel.end - channel.start} seconds")
        Duration: 3600.0 seconds
        >>> print(channel.end.iso_str)
        '1980-01-01T01:00:00.000000+00:00'
        """
        return self.start + ((self.n_samples - 1) / self.sample_rate)

    @property
    def sample_rate(self) -> float:
        """
        Get the sample rate in samples per second.

        Returns
        -------
        float
            Sample rate in Hz.

        Examples
        --------
        >>> print(f"Sample rate: {channel.sample_rate} Hz")
        Sample rate: 256.0 Hz
        """
        return self.metadata.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        """
        Set the sample rate with validation through metadata.

        Parameters
        ----------
        value : float
            New sample rate in Hz.

        Examples
        --------
        >>> channel.sample_rate = 256.0
        """
        self.metadata.sample_rate = value

    @property
    def n_samples(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of data points in the time series.

        Examples
        --------
        >>> print(f"Total samples: {channel.n_samples:,}")
        Total samples: 921,600
        >>> duration = channel.n_samples / channel.sample_rate
        >>> print(f"Duration: {duration/3600:.1f} hours")
        Duration: 1.0 hours
        """
        return self.hdf5_dataset.size

    @property
    def time_index(self) -> pd.DatetimeIndex:
        """
        Create a time index for the dataset based on metadata.

        Returns
        -------
        pd.DatetimeIndex
            Pandas datetime index spanning the entire dataset.

        Notes
        -----
        The time index is useful for time-based queries and slicing operations.
        It is generated dynamically from start time, sample rate, and number of samples.

        Examples
        --------
        >>> time_idx = channel.time_index
        >>> print(time_idx[0], time_idx[-1])
        1980-01-01 00:00:00 1980-01-01 00:59:59.996093750
        >>> print(f"Index length: {len(time_idx)}")
        Index length: 921600
        """
        return make_dt_coordinates(self.start, self.sample_rate, self.n_samples)

    def read_metadata(self) -> None:
        """
        Read metadata from HDF5 attributes into the metadata container.

        Loads all HDF5 attributes from the dataset and converts them to the
        appropriate Python types before populating the metadata object.

        For older MTH5 files, this method attempts to coerce values to the
        expected types based on the metadata schema to maintain backwards
        compatibility.

        Notes
        -----
        This method automatically validates metadata through the metadata
        container's validators. Type coercion is applied to handle older
        file formats that may have stored metadata with different types.

        Examples
        --------
        >>> channel.read_metadata()
        >>> print(channel.metadata.component)
        'Ex'
        >>> print(channel.metadata.sample_rate)
        256.0

        Handles type coercion for older files

        >>> # If sample_rate was stored as string '256.0' in old file
        >>> channel.read_metadata()
        >>> print(type(channel.metadata.sample_rate))
        <class 'float'>
        """
        meta_dict = read_attrs_to_dict(dict(self.hdf5_dataset.attrs), self.metadata)
        # Defensive check: skip if meta_dict is empty
        if not meta_dict:
            self.logger.debug(
                f"No metadata found for {self._class_name}, skipping from_dict."
            )
            return
        self.metadata.from_dict({self._class_name: meta_dict})
        self._has_read_metadata = True

    def write_metadata(self) -> None:
        """
        Write metadata from the container to HDF5 dataset attributes.

        Converts all metadata values to numpy-compatible types before writing
        to HDF5 attributes. Falls back to string conversion if direct conversion fails.

        Notes
        -----
        This method is automatically called during initialization and when
        metadata is updated.

        Examples
        --------
        >>> channel.metadata.component = 'Ey'
        >>> channel.metadata.measurement_azimuth = 90.0
        >>> channel.write_metadata()
        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            try:
                value = to_numpy_type(value)
                self.hdf5_dataset.attrs.create(key, value)
            except Exception as e:
                # Convert problematic values to string as fallback
                self.hdf5_dataset.attrs.create(key, str(value))

    def replace_dataset(self, new_data_array: np.ndarray) -> None:
        """
        Replace the entire dataset with new data.

        Parameters
        ----------
        new_data_array : np.ndarray
            New data array with shape (npts,). Must be 1-dimensional.

        Raises
        ------
        TypeError
            If new_data_array cannot be converted to numpy array.

        Notes
        -----
        The HDF5 dataset will be resized if the new array has a different shape.
        All existing data will be overwritten.

        Examples
        --------
        Replace with synthetic data

        >>> import numpy as np
        >>> new_data = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2560))
        >>> channel.replace_dataset(new_data)
        >>> print(f"New shape: {channel.hdf5_dataset.shape}")
        New shape: (2560,)

        Replace with processed data

        >>> original = channel.hdf5_dataset[:]
        >>> filtered = np.convolve(original, np.ones(5)/5, mode='same')
        >>> channel.replace_dataset(filtered)
        """
        if not isinstance(new_data_array, np.ndarray):
            try:
                new_data_array = np.array(new_data_array)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_data_array)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if new_data_array.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_data_array.shape)
        self.hdf5_dataset[...] = new_data_array

    def extend_dataset(
        self,
        new_data_array: np.ndarray,
        start_time: str | MTime,
        sample_rate: float,
        fill: str | float | int | None = None,
        max_gap_seconds: float | int = 1,
        fill_window: int = 10,
    ) -> None:
        """
        Extend or prepend data to the existing dataset with gap handling.

        Intelligently adds new data before, after, or within the existing time series.
        Handles time alignment, overlaps, and gaps with configurable fill strategies.

        Parameters
        ----------
        new_data_array : np.ndarray
            New data array with shape (npts,).
        start_time : str or MTime
            Start time of the new data array in UTC.
        sample_rate : float
            Sample rate of the new data array in Hz. Must match existing sample rate.
        fill : str, float, int, or None, optional
            Strategy for filling data gaps:

            - None : Raise MTH5Error if gap exists (default)
            - 'mean' : Fill with mean of both datasets within fill_window
            - 'median' : Fill with median of both datasets within fill_window
            - 'nan' : Fill with NaN values
            - numeric value : Fill with specified constant

        max_gap_seconds : float or int, optional
            Maximum allowed gap in seconds. Exceeding this raises MTH5Error.
            Default is 1 second.
        fill_window : int, optional
            Number of points from each dataset edge to estimate fill values.
            Default is 10 points.

        Raises
        ------
        MTH5Error
            If sample rates don't match, gap exceeds max_gap_seconds, or
            fill strategy is invalid.
        TypeError
            If new_data_array cannot be converted to numpy array.

        Notes
        -----
        - **Prepend**: New data start < existing start
        - **Append**: New data start > existing end
        - **Overwrite**: New data overlaps existing data

        The dataset is automatically resized to accommodate new data.

        Examples
        --------
        Append data with a small gap

        >>> ex = mth5_obj.get_channel('MT001', 'MT001a', 'Ex')
        >>> print(f"Original: {ex.n_samples} samples, ends {ex.end}")
        Original: 4096 samples, ends 2015-01-08T19:32:09.500000+00:00
        >>> new_data = np.random.randn(4096)
        >>> new_start = (ex.end + 0.5).isoformat()  # 0.5s gap
        >>> ex.extend_dataset(new_data, new_start, ex.sample_rate,
        ...                   fill='median', max_gap_seconds=2)
        >>> print(f"Extended: {ex.n_samples} samples, ends {ex.end}")
        Extended: 8200 samples, ends 2015-01-08T19:40:42.500000+00:00

        Prepend data seamlessly

        >>> prepend_data = np.random.randn(2048)
        >>> prepend_start = (ex.start - 2048/ex.sample_rate).isoformat()
        >>> ex.extend_dataset(prepend_data, prepend_start, ex.sample_rate)
        >>> print(f"New start: {ex.start}")

        Overwrite section of existing data

        >>> replacement_data = np.zeros(1024)
        >>> replace_start = (ex.start + 1.0).isoformat()  # 1s after start
        >>> ex.extend_dataset(replacement_data, replace_start, ex.sample_rate)

        """
        fw = fill_window
        # check input parameters
        if sample_rate != self.sample_rate:
            msg = (
                "new data must have the same sampling rate as existing data.\n"
                + f"\tnew sample rate =      {sample_rate}\n"
                + f"\texisting sample rate = {self.sample_rate}"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)
        if not isinstance(new_data_array, np.ndarray):
            try:
                new_data_array = np.array(new_data_array)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_data_array)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if not isinstance(start_time, MTime):
            start_time = MTime(time_stamp=start_time)
        # get end time will need later
        end_time = start_time + (new_data_array.size / sample_rate)

        # check start time
        start_t_diff = self._get_diff_new_array_start(start_time)
        end_t_diff = self._get_diff_new_array_end(end_time)

        self.logger.info("Extending data.")
        self.logger.info(f"Existing start time {self.start}")
        self.logger.info(f"New start time      {start_time}")
        self.logger.info(f"Existing end time   {self.end}")
        self.logger.info(f"New end time        {end_time}")

        # prepend data
        if start_t_diff < 0:
            self.logger.info("Prepending: ")
            self.logger.info(
                f"new start time {start_time} is before existing {self.start}"
            )
            if end_time.iso_no_tz not in self.time_index:
                gap = abs(end_time - self.start)
                if gap > 0:
                    if gap > max_gap_seconds:
                        msg = (
                            f"Time gap of {gap} seconds "
                            + f"is more than max_gap_seconds = {max_gap_seconds}."
                            + " Consider making a new run."
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    if fill is None:
                        msg = (
                            f"A time gap of {gap} seconds is found "
                            + "between new and existing data sets. \n"
                            + f"\tnew end time:        {end_time}\n"
                            + f"\texisting start time: {self.start}"
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    # set new start time
                    old_slice = self.time_slice(self.start, end_time=self.end)
                    old_start = self.start.copy()
                    self.start = start_time

                    # resize the existing data to make room for new data
                    self.hdf5_dataset.resize(
                        (
                            int(
                                new_data_array.size
                                + self.hdf5_dataset.size
                                + gap * sample_rate
                            ),
                        )
                    )

                    # fill based on time, refill existing data first
                    self.hdf5_dataset[
                        self.get_index_from_time(old_start) :
                    ] = old_slice.ts.values
                    self.hdf5_dataset[
                        0 : self.get_index_from_time(end_time)
                    ] = new_data_array

                    if fill == "mean":
                        fill_value = np.mean(
                            np.array(
                                [
                                    new_data_array[-fw:].mean(),
                                    float(old_slice.ts[0:fw].mean()),
                                ]
                            )
                        )
                    elif fill == "median":
                        fill_value = np.median(
                            np.array(
                                [
                                    np.median(new_data_array[-fw:]),
                                    np.median(old_slice.ts[0:fw]),
                                ]
                            )
                        )
                    elif fill == "nan":
                        fill_value = np.nan
                    elif isinstance(fill, (int, float)):
                        fill_value = fill
                    else:
                        msg = f"fill value {fill} is not understood"
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    self.logger.info(f"filling data gap with {fill_value}")
                    self.hdf5_dataset[
                        self.get_index_from_time(end_time) : self.get_index_from_time(
                            old_start
                        )
                    ] = fill_value
            else:
                new_size = (self.n_samples + int(abs(start_t_diff) * sample_rate),)
                overlap = abs(end_time - self.start)
                self.logger.warning(
                    f"New data is overlapping by {overlap} s."
                    + " Any overlap will be overwritten."
                )
                # set new start time
                old_slice = self.time_slice(self.start, end_time=self.end)
                old_start = self.start.copy()
                self.start = start_time
                self.logger.debug(
                    f"resizing data set from {self.n_samples} to {new_size}"
                )
                self.hdf5_dataset.resize(new_size)

                # put back the existing data, which any overlapping times
                # will be overwritten
                self.hdf5_dataset[
                    self.get_index_from_time(old_start) :
                ] = old_slice.ts.values
                self.hdf5_dataset[
                    0 : self.get_index_from_time(end_time)
                ] = new_data_array
        # append data
        elif start_t_diff > 0:
            old_end = self.end.copy()
            if start_time.iso_no_tz not in self.time_index:
                gap = abs(self.end - start_time)
                if gap > 0:
                    if gap > max_gap_seconds:
                        msg = (
                            f"Time gap of {gap} seconds "
                            + f"is more than max_gap_seconds = {max_gap_seconds}."
                            + " Consider making a new run."
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    if fill is None:
                        msg = (
                            f"A time gap of {gap} seconds is found "
                            + "between new and existing data sets. \n"
                            + f"\tnew start time:        {start_time}\n"
                            + f"\texisting end time:     {self.end}"
                        )
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    # resize the existing data to make room for new data
                    self.hdf5_dataset.resize(
                        (
                            int(
                                new_data_array.size
                                + self.hdf5_dataset.size
                                + gap * sample_rate
                            ),
                        )
                    )

                    self.hdf5_dataset[
                        self.get_index_from_time(start_time) :
                    ] = new_data_array
                    old_index = self.get_index_from_time(old_end)
                    if fill == "mean":
                        fill_value = np.mean(
                            np.array(
                                [
                                    new_data_array[0:fw].mean(),
                                    np.mean(self.hdf5_dataset[old_index - fw :]),
                                ]
                            )
                        )
                    elif fill == "median":
                        fill_value = np.median(
                            np.array(
                                [
                                    np.median(new_data_array[0:fw]),
                                    np.median(self.hdf5_dataset[old_index - fw :]),
                                ]
                            )
                        )
                    elif fill == "nan":
                        fill_value = np.nan
                    elif isinstance(fill, (int, float)):
                        fill_value = fill
                    else:
                        msg = f"fill value {fill} is not understood"
                        self.logger.error(msg)
                        raise MTH5Error(msg)
                    self.logger.info(f"filling data gap with {fill_value}")
                    self.hdf5_dataset[
                        self.get_index_from_time(old_end) : self.get_index_from_time(
                            start_time
                        )
                    ] = fill_value
            else:
                # if the new data fits within the extisting time span
                if end_t_diff < 0:
                    self.logger.debug(
                        "New data fits within existing time span"
                        + " all data in the window : "
                        f"{start_time} -- {end_time} " + "will be overwritten."
                    )
                    self.hdf5_dataset[
                        self.get_index_from_time(start_time) : self.get_index_from_time(
                            end_time
                        )
                    ] = new_data_array
                else:
                    new_size = (self.n_samples + int(abs(start_t_diff) * sample_rate),)
                    overlap = abs(self.end - start_time)
                    self.logger.warning(
                        f"New data is overlapping by {overlap} s."
                        + " Any overlap will be overwritten."
                    )

                    self.logger.debug(
                        f"resizing data set from {self.n_samples} to {new_size}"
                    )
                    self.hdf5_dataset.resize(new_size)

                    # put back the existing data, which any overlapping times
                    # will be overwritten
                    self.hdf5_dataset[
                        self.get_index_from_time(start_time) :
                    ] = new_data_array

    def has_data(self) -> bool:
        """
        Check if the channel contains non-zero data.

        Returns
        -------
        bool
            True if dataset has non-zero values, False if all zeros or empty.

        Examples
        --------
        >>> if channel.has_data():
        ...     print("Channel has valid data")
        ... else:
        ...     print("Channel is empty or all zeros")
        Channel has valid data

        >>> empty_channel.has_data()
        False
        """
        if len(self.hdf5_dataset) > 0:
            if len(np.nonzero(self.hdf5_dataset)[0]) > 0:
                return True
            else:
                return False
        return False

    def to_channel_ts(self) -> ChannelTS:
        """
        Convert the dataset to a ChannelTS object with full metadata.

        Returns
        -------
        ChannelTS
            Time series object with data, metadata, and channel response.

        Notes
        -----
        Data is loaded into memory. The resulting ChannelTS object is independent
        of the HDF5 file and can be modified without affecting the original dataset.

        Examples
        --------
        >>> ts = channel.to_channel_ts()
        >>> print(f"Type: {type(ts)}")
        Type: <class 'mt_timeseries.channel_ts.ChannelTS'>
        >>> print(f"Shape: {ts.ts.shape}, Mean: {ts.ts.mean():.2f}")
        Shape: (4096,), Mean: 0.15

        Process the time series

        >>> filtered_ts = ts.low_pass_filter(cutoff=10.0)
        >>> detrended_ts = ts.detrend('linear')
        >>> ts.plot()
        """
        # Now that copy() method is robust, we can use direct copying
        return ChannelTS(
            channel_type=self.metadata.type,
            data=self.hdf5_dataset[()],
            channel_metadata=self.metadata.copy(),
            run_metadata=self.run_metadata.copy(),
            station_metadata=self.station_metadata.copy(),
            survey_metadata=self.survey_metadata.copy(),
            channel_response=self.channel_response,
        )

    def to_xarray(self) -> xr.DataArray:
        """
        Convert the dataset to an xarray DataArray with time coordinates.

        Returns
        -------
        xr.DataArray
            DataArray with time index and metadata as attributes.

        Notes
        -----
        Data is loaded into memory. Metadata is stored in the attrs dictionary
        and will not be validated if modified.

        Examples
        --------
        >>> xr_data = channel.to_xarray()
        >>> print(xr_data)
        <xarray.DataArray (time: 4096)>
        array([0.931, 0.142, ..., 0.882])
        Coordinates:
          * time     (time) datetime64[ns] 1980-01-01 ... 1980-01-01T00:00:15.996
        Attributes:
            component:    Ex
            sample_rate:  256.0
            ...

        Use xarray's powerful selection

        >>> morning = xr_data.sel(time=slice('1980-01-01T06:00', '1980-01-01T12:00'))
        >>> daily_mean = xr_data.resample(time='1D').mean()
        >>> xr_data.plot()
        """
        return xr.DataArray(
            self.hdf5_dataset[()],
            coords=[("time", self.time_index)],
            attrs=self.metadata.to_dict(single=True),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame with time index.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'data' column and time index. Metadata stored in attrs.

        Notes
        -----
        Data is loaded into memory. Metadata is stored in the experimental
        attrs attribute and will not be validated if modified.

        Examples
        --------
        >>> df = channel.to_dataframe()
        >>> print(df.head())
                             data
        time
        1980-01-01 00:00:00  0.931
        1980-01-01 00:00:00  0.142
        ...

        Use pandas operations

        >>> df['data'].describe()
        >>> df.resample('1H').mean()
        >>> df.plot(y='data', figsize=(12, 4))

        Access metadata

        >>> print(df.attrs['component'])
        'Ex'
        >>> print(df.attrs['sample_rate'])
        256.0
        """
        df = pd.DataFrame({"data": self.hdf5_dataset[()]}, index=self.time_index)
        df.attrs.update(self.metadata.to_dict(single=True))

        return df

    def to_numpy(self) -> np.recarray:
        """
        Convert the dataset to a numpy structured array with time and data columns.

        Returns
        -------
        np.recarray
            Record array with 'time' and 'channel_data' fields.

        Notes
        -----
        Data is loaded into memory. The 'data' name is avoided as it's a
        builtin to numpy.

        Examples
        --------
        >>> arr = channel.to_numpy()
        >>> print(arr.dtype.names)
        ('time', 'channel_data')
        >>> print(arr['time'][0])
        1980-01-01T00:00:00.000000000
        >>> print(arr['channel_data'].mean())
        0.152

        Access fields

        >>> times = arr['time']
        >>> data = arr['channel_data']
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(times, data)
        """
        return np.core.records.fromarrays(
            [self.time_index.to_numpy(), self.hdf5_dataset[()]],
            names="time,channel_data",
        )

    def from_channel_ts(
        self,
        channel_ts_obj: ChannelTS,
        how: str = "replace",
        fill: str | float | int | None = None,
        max_gap_seconds: float | int = 1,
        fill_window: int = 10,
    ) -> None:
        """
        Populate the dataset from a ChannelTS object.

        Parameters
        ----------
        channel_ts_obj : ChannelTS
            Time series object containing data and metadata.
        how : {'replace', 'extend'}, optional
            Method for adding data:

            - 'replace' : Replace entire dataset (default)
            - 'extend' : Append/prepend to existing data with gap handling

        fill : str, float, int, or None, optional
            Gap filling strategy (only used with how='extend'):

            - None : Raise error on gaps (default)
            - 'mean' : Fill with mean of both datasets
            - 'median' : Fill with median of both datasets
            - 'nan' : Fill with NaN
            - numeric : Fill with constant value

        max_gap_seconds : float or int, optional
            Maximum allowed gap in seconds. Default is 1.
        fill_window : int, optional
            Points to use for estimating fill values. Default is 10.

        Raises
        ------
        TypeError
            If channel_ts_obj is not a ChannelTS instance.
        MTH5Error
            If time alignment or metadata validation fails.

        Examples
        --------
        Replace entire dataset

        >>> from mt_timeseries import ChannelTS
        >>> import numpy as np
        >>> ts = ChannelTS(
        ...     channel_type='electric',
        ...     data=np.random.randn(1000),
        ...     channel_metadata={'electric': {
        ...         'component': 'ex',
        ...         'sample_rate': 256.0
        ...     }}
        ... )
        >>> channel.from_channel_ts(ts, how='replace')
        >>> print(channel.n_samples)
        1000

        Extend existing dataset

        >>> new_ts = ChannelTS(
        ...     channel_type='electric',
        ...     data=np.random.randn(500),
        ...     channel_metadata={'electric': {
        ...         'component': 'ex',
        ...         'sample_rate': 256.0,
        ...         'time_period.start': channel.end.isoformat()
        ...     }}
        ... )
        >>> channel.from_channel_ts(new_ts, how='extend', fill='median')
        >>> print(channel.n_samples)
        1500
        """
        if not isinstance(channel_ts_obj, ChannelTS):
            msg = f"Input must be a ChannelTS object not {type(channel_ts_obj)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if how == "replace":
            # Get the metadata dict first to avoid deepcopy issues with HDF5 references
            # channel_ts_obj.channel_metadata.mth5_type = (
            #     channel_ts_obj.channel_metadata._class_name
            # )
            metadata_dict = channel_ts_obj.channel_metadata.to_dict()
            metadata_dict[channel_ts_obj.channel_metadata._class_name][
                "mth5_type"
            ] = channel_ts_obj.channel_metadata._class_name
            # metadata_dict[channel_ts_obj.channel_metadata._class_name][
            #     "hdf5_reference"
            # ] = self.hdf5_dataset.ref
            # Update metadata with MTH5-specific attributes
            self.metadata.from_dict(metadata_dict)
            self.metadata.mth5_type = self._class_name
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.replace_dataset(channel_ts_obj.ts)
            # # apparently need to reset these otherwise they get overwritten with None
            # self.metadata.hdf5_reference = self.hdf5_dataset.ref
            # self.metadata.mth5_type = self._class_name
            self.write_metadata()
        elif how == "extend":
            self.extend_dataset(
                channel_ts_obj.ts,
                channel_ts_obj.start,
                channel_ts_obj.sample_rate,
                fill=fill,
            )
        #
        # TODO need to check on metadata.

    def from_xarray(
        self,
        data_array: xr.DataArray,
        how: str = "replace",
        fill: str | float | int | None = None,
        max_gap_seconds: float | int = 1,
        fill_window: int = 10,
    ) -> None:
        """
        Populate the dataset from an xarray DataArray.

        Parameters
        ----------
        data_array : xr.DataArray
            DataArray with time coordinate and metadata in attrs.
        how : {'replace', 'extend'}, optional
            Method for adding data:

            - 'replace' : Replace entire dataset (default)
            - 'extend' : Append/prepend to existing data with gap handling

        fill : str, float, int, or None, optional
            Gap filling strategy (only used with how='extend'):

            - None : Raise error on gaps (default)
            - 'mean' : Fill with mean of both datasets
            - 'median' : Fill with median of both datasets
            - 'nan' : Fill with NaN
            - numeric : Fill with constant value

        max_gap_seconds : float or int, optional
            Maximum allowed gap in seconds. Default is 1.
        fill_window : int, optional
            Points to use for estimating fill values. Default is 10.

        Raises
        ------
        TypeError
            If data_array is not an xarray.DataArray.
        MTH5Error
            If time alignment fails.

        Examples
        --------
        Replace from xarray

        >>> import xarray as xr
        >>> import numpy as np
        >>> import pandas as pd
        >>> time = pd.date_range('2020-01-01', periods=1000, freq='0.004S')
        >>> data = xr.DataArray(
        ...     np.random.randn(1000),
        ...     coords=[('time', time)],
        ...     attrs={'component': 'ex', 'sample_rate': 256.0}
        ... )
        >>> channel.from_xarray(data, how='replace')
        >>> print(channel.n_samples)
        1000

        Extend from xarray with gap

        >>> time2 = pd.date_range('2020-01-01T00:00:05', periods=500, freq='0.004S')
        >>> data2 = xr.DataArray(np.random.randn(500), coords=[('time', time2)])
        >>> channel.from_xarray(data2, how='extend', fill='mean')
        """
        if not isinstance(data_array, xr.DataArray):
            msg = f"Input must be a xarray.DataArray object not {type(data_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if how == "replace":
            self.metadata.from_dict({self.metadata._class_name: data_array.attrs})
            self.replace_dataset(data_array.values)
            self.write_metadata()
        elif how == "extend":
            self.extend_dataset(
                data_array.values,
                data_array.coords.indexes["time"][0].isoformat(),
                1e9 / data_array.coords.indexes["time"][0].freq.nanos,
                fill=fill,
            )
        # TODO need to check on metadata.

    def _get_diff_new_array_start(self, start_time: str | MTime) -> float:
        """
        Calculate time difference between new array start and existing start.

        Parameters
        ----------
        start_time : str or MTime
            Start time of the new array.

        Returns
        -------
        float
            Time difference in seconds (new_start - existing_start).
            Positive means new starts later, negative means new starts earlier.

        Examples
        --------
        >>> diff = channel._get_diff_new_array_start('1980-01-01T00:05:00')
        >>> print(f"New data starts {diff} seconds after existing")
        New data starts 300.0 seconds after existing
        """
        if not isinstance(start_time, MTime):
            start_time = MTime(time_stamp=start_time)
        t_diff = 0
        if start_time != self.start:
            t_diff = start_time - self.start
        return t_diff

    def _get_diff_new_array_end(self, end_time: str | MTime) -> float:
        """
        Calculate time difference between new array end and existing end.

        Parameters
        ----------
        end_time : str or MTime
            End time of the new array.

        Returns
        -------
        float
            Time difference in seconds (new_end - existing_end).
            Positive means new ends later, negative means new ends earlier.

        Examples
        --------
        >>> diff = channel._get_diff_new_array_end('1980-01-01T00:05:00')
        >>> print(f"Difference: {diff} seconds")
        Difference: -3300.0 seconds
        """
        if not isinstance(end_time, MTime):
            end_time = MTime(time_stamp=end_time)
        t_diff = 0
        if end_time != self.end:
            t_diff = end_time - self.end
        return t_diff

    @property
    def channel_entry(self) -> np.ndarray:
        """
        Create a structured array entry for channel summary tables.

        Returns
        -------
        np.ndarray
            Structured array with dtype=CHANNEL_DTYPE containing channel metadata
            and HDF5 references for survey-wide summaries.

        Notes
        -----
        This entry includes survey ID, station ID, run ID, location, component,
        time period, sample rate, and HDF5 references for navigation.

        Examples
        --------
        >>> entry = channel.channel_entry
        >>> print(entry['component'][0])
        'Ex'
        >>> print(entry['sample_rate'][0])
        256.0
        >>> print(entry['station'][0])
        'MT001'
        """
        return np.array(
            [
                (
                    self.survey_id,
                    self.hdf5_dataset.parent.parent.attrs["id"],
                    self.hdf5_dataset.parent.attrs["id"],
                    self.hdf5_dataset.parent.parent.attrs["location.latitude"],
                    self.hdf5_dataset.parent.parent.attrs["location.longitude"],
                    self.hdf5_dataset.parent.parent.attrs["location.elevation"],
                    self.metadata.component,
                    self.metadata.time_period.start,
                    self.metadata.time_period.end,
                    self.hdf5_dataset.size,
                    self.metadata.sample_rate,
                    self.metadata.type,
                    self.metadata.measurement_azimuth,
                    self.metadata.measurement_tilt,
                    self.metadata.units,
                    self.hdf5_dataset.ref,
                    self.hdf5_dataset.parent.ref,
                    self.hdf5_dataset.parent.parent.ref,
                )
            ],
            dtype=CHANNEL_DTYPE,
        )

    def time_slice(
        self,
        start: str | MTime,
        end: str | MTime | None = None,
        n_samples: int | None = None,
        return_type: str = "channel_ts",
    ) -> ChannelTS | xr.DataArray | pd.DataFrame | np.ndarray:
        """
        Extract a time slice from the channel dataset.

        Parameters
        ----------
        start : str or MTime
            Start time of the slice in UTC.
        end : str or MTime, optional
            End time of the slice. Mutually exclusive with n_samples.
        n_samples : int, optional
            Number of samples to extract. Mutually exclusive with end.
        return_type : {'channel_ts', 'xarray', 'pandas', 'numpy'}, optional
            Format for returned data. Default is 'channel_ts'.

        Returns
        -------
        ChannelTS or xr.DataArray or pd.DataFrame or np.ndarray
            Time slice in the requested format with appropriate metadata.

        Raises
        ------
        ValueError
            If both end and n_samples are provided or neither is provided.

        Notes
        -----
        - If the requested slice extends beyond available data, it will be
          automatically truncated with a warning.
        - Regional HDF5 references are used when possible for efficiency.

        Examples
        --------
        Extract by number of samples

        >>> ex = mth5_obj.get_channel('FL001', 'FL001a', 'Ex')
        >>> ex_slice = ex.time_slice(\"2015-01-08T19:49:15\", n_samples=4096)
        >>> print(type(ex_slice))
        <class 'mt_timeseries.channel_ts.ChannelTS'>
        >>> print(f\"Slice shape: {ex_slice.ts.shape}\")\n        Slice shape: (4096,)
        >>> ex_slice.plot()

        Extract by time range

        >>> ex_slice = ex.time_slice(\n        ...     \"2015-01-08T19:49:15\",
        ...     end=\"2015-01-08T20:49:15\"\n        ... )
        >>> print(f\"Duration: {ex_slice.end - ex_slice.start} seconds\")
        Duration: 3600.0 seconds

        Return as xarray for analysis

        >>> xr_slice = ex.time_slice(\n        ...     \"2015-01-08T19:49:15\",
        ...     n_samples=1000,
        ...     return_type='xarray'\n        ... )
        >>> print(xr_slice.mean().values)
        0.152
        >>> xr_slice.plot()

        Return as pandas for tabular ops

        >>> df_slice = ex.time_slice(\n        ...     \"2015-01-08T19:49:15\",
        ...     n_samples=500,
        ...     return_type='pandas'\n        ... )
        >>> df_slice['data'].describe()
        >>> df_slice.resample('10S').mean()

        Return as numpy for computation

        >>> np_slice = ex.time_slice(\n        ...     \"2015-01-08T19:49:15\",
        ...     n_samples=100,
        ...     return_type='numpy'\n        ... )
        >>> np.fft.fft(np_slice)

        """

        start_index, end_index, npts = self._get_slice_index_values(
            start, end, n_samples
        )
        if npts > self.hdf5_dataset.size or end_index > self.hdf5_dataset.size:
            # leave as +1 to be inclusive
            end_index = self.hdf5_dataset.size
            msg = (
                "Requested slice is larger than data.  "
                f"Slice length = {npts}, data length = {self.hdf5_dataset.shape}. "
                f"Setting end_index to {end_index}"
            )

            self.logger.warning(msg)
        if start_index < 0:
            # leave as +1 to be inclusive
            start_index = 0

            msg = (
                f"Requested start {start} is before data start {self.start}. "
                f"Setting start_index to {start_index} and start to {self.start}"
            )

            self.logger.warning(msg)
            start = self.start

        # create a regional reference that can be used
        try:
            regional_ref = self.hdf5_dataset.regionref[start_index:end_index]
        except (OSError, RuntimeError):
            self.logger.debug(
                "file is in read mode cannot set an internal reference, using index values"
            )
            regional_ref = slice(start_index, end_index)
        dt_index = make_dt_coordinates(start, self.sample_rate, npts)

        meta_dict = self.metadata.to_dict()[self.metadata._class_name]
        meta_dict["time_period.start"] = dt_index[0].isoformat()
        meta_dict["time_period.end"] = dt_index[-1].isoformat()

        data = None
        if return_type == "xarray":
            # need the +1 to be inclusive of the last point
            data = xr.DataArray(
                self.hdf5_dataset[regional_ref], coords=[("time", dt_index)]
            )
            data.attrs.update(meta_dict)
        elif return_type == "pandas":
            data = pd.DataFrame(
                {"data": self.hdf5_dataset[regional_ref]}, index=dt_index
            )
            data.attrs.update(meta_dict)
        elif return_type == "numpy":
            data = self.hdf5_dataset[regional_ref]
        elif return_type == "channel_ts":
            data = ChannelTS(
                self.metadata.type,
                data=self.hdf5_dataset[regional_ref],
                survey_metadata=self.survey_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                channel_metadata={self.metadata.type: meta_dict},
                channel_response=self.channel_response,
            )
        else:
            msg = "return_type not understood, must be [ pandas | numpy | channel_ts ]"
            self.logger.error(msg)
            raise ValueError(msg)
        return data

    def get_index_from_time(self, given_time: str | MTime) -> int:
        """
        Calculate the array index for a given time.

        Parameters
        ----------
        given_time : str or MTime
            Time to convert to index.

        Returns
        -------
        int
            Array index corresponding to the given time.

        Notes
        -----
        Index is calculated as: (time - start_time) * sample_rate
        and rounded to nearest integer.

        Examples
        --------
        >>> idx = channel.get_index_from_time('1980-01-01T00:00:10')
        >>> print(f"Index for 10 seconds: {idx}")
        Index for 10 seconds: 2560
        >>> # With 256 Hz sample rate: 10 * 256 = 2560

        >>> start_idx = channel.get_index_from_time(channel.start)
        >>> print(start_idx)
        0
        """
        if not isinstance(given_time, MTime):
            given_time = MTime(time_stamp=given_time)
        index = (
            given_time - self.metadata.time_period.start
        ) * self.metadata.sample_rate

        return int(round(index))

    def get_index_from_end_time(self, given_time: str | MTime) -> int:
        """
        Get the end index value (inclusive) for a given time.

        Parameters
        ----------
        given_time : str or MTime
            Time to convert to end index.

        Returns
        -------
        int
            Array index + 1 for inclusive slicing.

        Notes
        -----
        Adds 1 to the calculated index to make it suitable for
        inclusive end slicing (e.g., array[start:end]).

        Examples
        --------
        >>> end_idx = channel.get_index_from_end_time('1980-01-01T00:00:10')
        >>> data_slice = channel.hdf5_dataset[0:end_idx]
        >>> # Includes sample at exactly 10 seconds
        """
        return self.get_index_from_time(given_time) + 1

    def _get_slice_index_values(
        self,
        start: str | MTime,
        end: str | MTime | None = None,
        n_samples: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Calculate start index, end index, and number of points for a time slice.

        Parameters
        ----------
        start : str or MTime
            Start time of the slice.
        end : str or MTime, optional
            End time of the slice. Mutually exclusive with n_samples.
        n_samples : int, optional
            Number of samples. Mutually exclusive with end.

        Returns
        -------
        tuple of (int, int, int)
            (start_index, end_index, n_points) for array slicing.

        Raises
        ------
        ValueError
            If both end and n_samples are provided or neither is provided.

        Examples
        --------
        >>> start_idx, end_idx, npts = channel._get_slice_index_values(
        ...     '1980-01-01T00:00:05',
        ...     n_samples=1000
        ... )
        >>> print(f"Indices: {start_idx} to {end_idx}, {npts} points")
        Indices: 1280 to 2280, 1000 points

        >>> start_idx, end_idx, npts = channel._get_slice_index_values(
        ...     '1980-01-01T00:00:00',
        ...     end='1980-01-01T00:00:10'
        ... )
        >>> print(f"10 second slice: {npts} points")
        10 second slice: 2560 points
        """
        start = MTime(time_stamp=start)
        if end is not None:
            end = MTime(time_stamp=end)
        if n_samples is not None:
            n_samples = int(n_samples)

        if n_samples is None and end is None:
            msg = "Must input either end_time or n_samples."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_samples is not None and end is not None:
            msg = "Must input either end_time or n_samples, not both."
            self.logger.error(msg)
            raise ValueError(msg)

        # if end time is given
        if end is not None and n_samples is None:
            start_index = self.get_index_from_time(start)
            end_index = self.get_index_from_end_time(end)
            npts = int(end_index - start_index)
        # if n_samples are given
        elif end is None and n_samples is not None:
            start_index = self.get_index_from_time(start)
            # leave as +1 to be inclusive
            end_index = start_index + n_samples
            npts = n_samples

        return start_index, end_index, npts


@inherit_doc_string
class ElectricDataset(ChannelDataset):
    """
    Specialized container for electric field channel data.

    Inherits all functionality from ChannelDataset with electric field
    specific metadata handling.

    Parameters
    ----------
    group : h5py.Dataset
        HDF5 dataset containing electric field data.
    **kwargs : dict
        Additional keyword arguments passed to ChannelDataset.

    Examples
    --------
    >>> ex_dataset = run_group.get_channel('Ex')
    >>> print(type(ex_dataset))
    <class 'mth5.groups.channel_dataset.ElectricDataset'>
    >>> print(ex_dataset.metadata.type)
    'electric'
    >>> print(ex_dataset.metadata.units)
    'mV/km'
    """

    def __init__(self, group: h5py.Dataset, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)


@inherit_doc_string
class MagneticDataset(ChannelDataset):
    """
    Specialized container for magnetic field channel data.

    Inherits all functionality from ChannelDataset with magnetic field
    specific metadata handling.

    Parameters
    ----------
    group : h5py.Dataset
        HDF5 dataset containing magnetic field data.
    **kwargs : dict
        Additional keyword arguments passed to ChannelDataset.

    Examples
    --------
    >>> hx_dataset = run_group.get_channel('Hx')
    >>> print(type(hx_dataset))
    <class 'mth5.groups.channel_dataset.MagneticDataset'>
    >>> print(hx_dataset.metadata.type)
    'magnetic'
    >>> print(hx_dataset.metadata.units)
    'nT'
    """

    def __init__(self, group: h5py.Dataset, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)


@inherit_doc_string
class AuxiliaryDataset(ChannelDataset):
    """
    Specialized container for auxiliary channel data.

    Inherits all functionality from ChannelDataset with auxiliary channel
    specific metadata handling. Used for temperature, battery voltage, etc.

    Parameters
    ----------
    group : h5py.Dataset
        HDF5 dataset containing auxiliary data.
    **kwargs : dict
        Additional keyword arguments passed to ChannelDataset.

    Examples
    --------
    >>> temp_dataset = run_group.get_channel('Temperature')
    >>> print(type(temp_dataset))
    <class 'mth5.groups.channel_dataset.AuxiliaryDataset'>
    >>> print(temp_dataset.metadata.type)
    'auxiliary'
    >>> print(temp_dataset.metadata.units)
    'celsius'
    """

    def __init__(self, group: h5py.Dataset, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)
