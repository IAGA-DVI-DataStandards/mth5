# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:59:03 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import inspect
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
from mt_metadata import timeseries as metadata
from mt_timeseries import ChannelTS, RunTS

from mth5 import CHUNK_SIZE
from mth5.groups import (
    AuxiliaryDataset,
    BaseGroup,
    ChannelDataset,
    ElectricDataset,
    MagneticDataset,
)
from mth5.helpers import read_attrs_to_dict, to_numpy_type, validate_name
from mth5.utils.exceptions import MTH5Error


meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# =============================================================================


# =============================================================================
# Run Group
# =============================================================================
class RunGroup(BaseGroup):
    """
    Container for a single MT measurement run with multiple channels.

    Manages time series data and metadata for one measurement run within a station.
    A run can contain multiple channels of electric, magnetic, and auxiliary data.
    This class provides methods to add, retrieve, and manage individual channels,
    along with convenient access to station and survey metadata.

    The run group is located at ``/Survey/Stations/{station_name}/{run_name}`` in
    the HDF5 file hierarchy.

    Attributes
    ----------
    metadata : mt_metadata.timeseries.Run
        Run metadata including sample rate, time period, and channel information.
    channel_summary : pd.DataFrame
        Summary table of all channels in the run.
    groups_list : list[str]
        List of channel names in the run.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group for the run, should have path like
        ``/Survey/Stations/{station_name}/{run_name}``
    run_metadata : mt_metadata.timeseries.Run, optional
        Metadata container for the run. Default is None.
    **kwargs : Any
        Additional keyword arguments passed to BaseGroup.

    Notes
    -----
    Key behaviors:

    - Channels can be of type: electric, magnetic, or auxiliary
    - All metadata updates should use the metadata object for validation
    - Call write_metadata() after modifying metadata to persist changes
    - Channel metadata is cached for performance during repeated access
    - Deleting a channel removes the reference but doesn't reduce file size

    Examples
    --------
    Access run from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

    Check available channels:

    >>> run.groups_list
    ['Ex', 'Ey', 'Hx', 'Hy']

    Access HDF5 group directly:

    >>> run.hdf5_group.ref
    <HDF5 Group Reference>

    Update metadata and persist to file:

    >>> run.metadata.sample_rate = 512.0
    >>> run.write_metadata()

    Add a channel:

    >>> import numpy as np
    >>> data = np.random.rand(4096)
    >>> ex = run.add_channel('Ex', 'electric', data=data)

    This class provides methods to add and get channels.  A summary table of
    all existing channels in the run is also provided as a convenience look up
    table to make searching easier.

    :param group: HDF5 group for a station, should have a path
                  ``/Survey/Stations/station_name/run_name``
    :type group: :class:`h5py.Group`
    :param station_metadata: metadata container, defaults to None
    :type station_metadata: :class:`mth5.metadata.Station`, optional

    :Access RunGroup from an open MTH5 file:

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

    :Check what channels exist:

    >>> station.groups_list
    ['Ex', 'Ey', 'Hx', 'Hy']

    To access the hdf5 group directly use `RunGroup.hdf5_group`

    >>> station.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `SurveyGroup.write_metadata()` method.  This is a temporary
             solution, working on an automatic updater if metadata is changed.

    >>> run.metadata.existing_attribute = 'update_existing_attribute'
    >>> run.write_metadata()

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> station.metadata.add_base_attribute('new_attribute',
    >>> ...                                 'new_attribute_value',
    >>> ...                                 {'type':str,
    >>> ...                                  'required':True,
    >>> ...                                  'style':'free form',
    >>> ...                                  'description': 'new attribute desc.',
    >>> ...                                  'units':None,
    >>> ...                                  'options':[],
    >>> ...                                  'alias':[],
    >>> ...                                  'example':'new attribute

    :Add a channel:

    >>> new_channel = run.add_channel('Ex', 'electric',
    >>> ...                            data=numpy.random.rand(4096))
    >>> new_run
    /Survey/Stations/MT001/MT001a:
    =======================================
        --> Dataset: summary
        ......................
        --> Dataset: Ex
        ......................
        --> Dataset: Ey
        ......................
        --> Dataset: Hx
        ......................
        --> Dataset: Hy
        ......................


    :Add a channel with metadata:

    >>> from mth5.metadata import Electric
    >>> ex_metadata = Electric()
    >>> ex_metadata.time_period.start = '2020-01-01T12:30:00'
    >>> ex_metadata.time_period.end = '2020-01-03T16:30:00'
    >>> new_ex = run.add_channel('Ex', 'electric',
    >>> ...                       channel_metadata=ex_metadata)
    >>> # to look at the metadata
    >>> new_ex.metadata
    {
         "electric": {
            "ac.end": 1.2,
            "ac.start": 2.3,
            ...
            }
    }


    .. seealso:: `mth5.metadata` for details on how to add metadata from
                 various files and python objects.

    :Remove a channel:

    >>> run.remove_channel('Ex')
    >>> station
    /Survey/Stations/MT001/MT001a:
    =======================================
        --> Dataset: summary
        ......................
        --> Dataset: Ey
        ......................
        --> Dataset: Hx
        ......................
        --> Dataset: Hy
        ......................

    .. note:: Deleting a station is not as simple as del(station).  In HDF5
              this does not free up memory, it simply removes the reference
              to that station.  The common way to get around this is to
              copy what you want into a new file, or overwrite the station.

    :Get a channel:

    >>> existing_ex = stations.get_channel('Ex')
    >>> existing_ex
    Channel Electric:
    -------------------
        data type:        Ex
        data type:        electric
        data format:      float32
        data shape:       (4096,)
        start:            1980-01-01T00:00:00+00:00
        end:              1980-01-01T00:32:+08:00
        sample rate:      8


    :summary Table:

    A summary table is provided to make searching easier.  The table
    summarized all stations within a survey. To see what names are in the
    summary table:

    >>> run.summary_table.dtype.descr
    [('component', ('|S5', {'h5py_encoding': 'ascii'})),
     ('start', ('|S32', {'h5py_encoding': 'ascii'})),
     ('end', ('|S32', {'h5py_encoding': 'ascii'})),
     ('n_samples', '<i4'),
     ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
     ('units', ('|S25', {'h5py_encoding': 'ascii'})),
     ('hdf5_reference', ('|O', {'ref': h5py.h5r.Reference}))]


    .. note:: When a run is added an entry is added to the summary table,
              where the information is pulled from the metadata.

    >>> new_run.summary_table
    index | component | start | end | n_samples | measurement_type | units |
    hdf5_reference
    --------------------------------------------------------------------------
    -------------
    """

    def __init__(
        self,
        group: h5py.Group,
        run_metadata: Optional[metadata.Run] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize RunGroup.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group for the run.
        run_metadata : mt_metadata.timeseries.Run, optional
            Metadata container for the run. Default is None.
        **kwargs : Any
            Additional keyword arguments passed to BaseGroup.
        """
        self._non_channel_groups = ["Features"]
        super().__init__(group, group_metadata=run_metadata, **kwargs)
        # Channel metadata cache to share objects between add_channel and metadata property
        self._channel_metadata_cache: dict[
            str, metadata.Electric | metadata.Magnetic | metadata.Auxiliary
        ] = {}

    @property
    def station_metadata(self) -> metadata.Station:
        """
        Get station metadata with current run included.

        Returns
        -------
        metadata.Station
            Station metadata object containing this run's information.

        Examples
        --------
        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5("example.h5", mode='r')
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> station_meta = run.station_metadata
        >>> print(station_meta.id)
        MT001
        """
        meta_dict = dict(self.hdf5_group.parent.attrs)
        meta_dict["run_list"] = [self.metadata.id]
        station_metadata = metadata.Station()
        meta_dict = read_attrs_to_dict(meta_dict, metadata.Station())
        station_metadata.from_dict({"station": meta_dict})
        station_metadata.add_run(self.metadata)

        return station_metadata

    @property
    def survey_metadata(self) -> metadata.Survey:
        """
        Get survey metadata with current station and run included.

        Returns
        -------
        metadata.Survey
            Survey metadata object containing the full hierarchy.

        Examples
        --------
        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5("example.h5", mode='r')
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> survey_meta = run.survey_metadata
        >>> print(survey_meta.id)
        CONUS_South
        """
        meta_dict = read_attrs_to_dict(
            dict(self.hdf5_group.parent.parent.parent.attrs), metadata.Survey()
        )
        survey_metadata = metadata.Survey()
        survey_metadata.from_dict({"survey": meta_dict})
        survey_metadata.add_station(self.station_metadata)
        return survey_metadata

    def _read_channel_metadata_from_hdf5(
        self, channel_name: str
    ) -> metadata.Electric | metadata.Magnetic | metadata.Auxiliary:
        """
        Read channel metadata from HDF5 and return metadata object.

        Parameters
        ----------
        channel_name : str
            Name of the channel to read metadata for.

        Returns
        -------
        metadata.Electric | metadata.Magnetic | metadata.Auxiliary
            Channel metadata object of appropriate type.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> ex_meta = run._read_channel_metadata_from_hdf5("ex")
        >>> print(ex_meta.type)
        electric
        """
        meta_dict = dict(self.hdf5_group[channel_name].attrs)
        meta_dict = read_attrs_to_dict(
            meta_dict,
            meta_classes[meta_dict["type"].capitalize()](),
        )
        ch_metadata = meta_classes[meta_dict["type"].capitalize()]()
        ch_metadata.from_dict(meta_dict)
        return ch_metadata

    def recache_channel_metadata(self) -> None:
        """
        Clear and rebuild the channel metadata cache from current HDF5 data.

        This method reads all channel metadata from HDF5 storage and updates
        the internal cache. Useful when channel metadata has been modified
        externally or needs to be synchronized.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> run.recache_channel_metadata()
        >>> # Cache is now synchronized with HDF5 storage
        """
        self._channel_metadata_cache = {}
        for ch in self.groups_list:
            if ch in self._non_channel_groups:
                continue
            ch_metadata = self._read_channel_metadata_from_hdf5(ch)
            self._channel_metadata_cache[ch] = ch_metadata

    @BaseGroup.metadata.getter
    def metadata(self) -> metadata.Run:
        """
        Get run metadata including all channel information.

        This property dynamically reads and caches channel metadata from HDF5,
        ensuring the run metadata always reflects the current state of channels.

        Returns
        -------
        metadata.Run
            Run metadata object with all channels included.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> run_meta = run.metadata
        >>> print(run_meta.channels_recorded_electric)
        ['ex', 'ey']
        >>> print(run_meta.sample_rate)
        256.0
        """
        if not self._has_read_metadata:
            self.read_metadata()
            self._has_read_metadata = True

        if len(self._metadata.channels) > 0:
            if (
                self._metadata.time_period.start
                != self._metadata.channels[0].time_period.start
            ) or (
                self._metadata.time_period.end
                != self._metadata.channels[0].time_period.end
            ):
                self.recache_channel_metadata()

                # Clear and rebuild the channels list
                self._metadata._empty_channels_recorded()
                self._metadata.channels = []

                for ch in self.groups_list:
                    if ch in self._non_channel_groups:
                        continue
                    if ch in self._channel_metadata_cache:
                        # Reuse cached metadata to prevent duplicate processing
                        cached_metadata = self._channel_metadata_cache[ch]
                        self._metadata.add_channel(cached_metadata)
                    else:
                        # Create new metadata if not cached
                        ch_metadata = self._read_channel_metadata_from_hdf5(ch)
                        # Cache the metadata for future use
                        self._channel_metadata_cache[ch] = ch_metadata
                        self._metadata.add_channel(ch_metadata)

        # Only rebuild channels if they haven't been built yet or if the group list has changed
        if not self._metadata.channels or len(self._metadata.channels) != len(
            self.groups_list
        ):
            # Get current channel names from the groups and existing channels
            current_group_names = set(self.groups_list)
            existing_channel_names = set(ch.component for ch in self._metadata.channels)

            # Only rebuild if there's actually a difference in the channel sets
            if current_group_names != existing_channel_names:
                # Clear and rebuild the channels list
                self._metadata._empty_channels_recorded()
                self._metadata.channels = []

                # List of known non-channel subgroups to skip
                for ch in self.groups_list:
                    # Skip non-channel groups
                    if ch in self._non_channel_groups:
                        continue
                    if ch in self._channel_metadata_cache:
                        # Reuse cached metadata to prevent duplicate processing
                        cached_metadata = self._channel_metadata_cache[ch]
                        self._metadata.add_channel(cached_metadata)
                    else:
                        # Create new metadata if not cached
                        ch_metadata = self._read_channel_metadata_from_hdf5(ch)
                        # Cache the metadata for future use
                        self._channel_metadata_cache[ch] = ch_metadata
                        self._metadata.add_channel(ch_metadata)
            # If channel sets are identical, skip rebuilding to prevent duplicates
        self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def channel_summary(self) -> pd.DataFrame:
        """
        Get summary of all channels in the run as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: component, start, end, n_samples,
            sample_rate, measurement_type, units, hdf5_reference.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> summary = run.channel_summary
        >>> print(summary[['component', 'sample_rate', 'n_samples']])
          component  sample_rate  n_samples
        0        ex        256.0      65536
        1        ey        256.0      65536
        2        hx        256.0      65536
        3        hy        256.0      65536
        """
        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["type"]
                if ch_type in ["electric", "magnetic", "auxiliary"]:
                    ch_list.append(
                        (
                            group.attrs["component"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.size,
                            group.attrs["sample_rate"],
                            group.attrs["type"],
                            group.attrs["units"],
                            group.ref,
                        )
                    )
            except KeyError:
                pass
        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("n_samples", int),
                    ("sample_rate", float),
                    ("measurement_type", "U12"),
                    ("units", "U25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def write_metadata(self) -> None:
        """
        Write run metadata to HDF5 attributes.

        Converts metadata object to dictionary and writes all attributes
        to the HDF5 group.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> run.metadata.sample_rate = 512.0
        >>> run.write_metadata()
        >>> # Metadata is now persisted to HDF5 file
        """
        for key, value in self.metadata.to_dict(single=True).items():
            value = to_numpy_type(value)
            self.hdf5_group.attrs.create(key, value)

    def add_channel(
        self,
        channel_name,
        channel_type,
        data,
        channel_dtype="int32",
        shape=None,
        max_shape=(None,),
        chunks=True,
        channel_metadata=None,
        **kwargs,
    ):
        """
        Add a channel to the run.

        Parameters
        ----------
        channel_name : str
            Name of the channel (e.g., 'ex', 'ey', 'hx', 'hy', 'hz').
        channel_type : str
            Type of channel: 'electric', 'magnetic', or 'auxiliary'.
        data : numpy.ndarray or None
            Time series data for the channel. If None, an empty resizable
            dataset will be created.
        channel_dtype : str, optional
            Data type for the channel if data is None, by default "int32".
        shape : tuple of int, optional
            Initial shape of the dataset. If None and data is None, shape
            is estimated from metadata or set to (1,), by default None.
        max_shape : tuple of int or None, optional
            Maximum shape the dataset can be resized to. Use None for
            unlimited growth in that dimension, by default (None,).
        chunks : bool or int, optional
            Enable chunked storage. If True, uses automatic chunking.
            If int, uses that chunk size, by default True.
        channel_metadata : mt_metadata.timeseries.Electric, Magnetic, or Auxiliary, optional
            Metadata object for the channel, by default None.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ElectricDataset or MagneticDataset or AuxiliaryDataset
            The created channel dataset object.

        Raises
        ------
        MTH5Error
            If channel_type is not one of: electric, magnetic, auxiliary.

        Examples
        --------
        Add a channel with data:

        >>> import numpy as np
        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5("example.h5", mode='a')
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> data = np.random.rand(4096)
        >>> ex = run.add_channel('ex', 'electric', data)
        >>> print(ex.metadata.component)
        ex

        Add a channel with metadata:

        >>> from mt_metadata.timeseries import Electric
        >>> ex_meta = Electric()
        >>> ex_meta.time_period.start = '2020-01-01T12:30:00'
        >>> ex_meta.sample_rate = 256.0
        >>> ex = run.add_channel('ex', 'electric', None,
        ...                      channel_metadata=ex_meta)
        >>> print(ex.metadata.sample_rate)
        256.0

        Add a channel with custom shape:

        >>> ex = run.add_channel('ex', 'electric', None,
        ...                      shape=(8192,), channel_dtype='float32')
        >>> print(ex.hdf5_dataset.shape)
        (8192,)
        """
        channel_name = validate_name(channel_name.lower())
        estimate_size = (1,)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if data is not None:
            if data.size < 1024:
                chunks = None
        try:
            if data is not None:
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    data=data,
                    dtype=data.dtype,
                    chunks=chunks,
                    maxshape=max_shape,
                    **self.dataset_options,
                )
            # initialize a resizable data array
            # need to set the chunk size to something useful, if the chunk
            # size is 1 this causes performance issues and bloating of the
            # hdf5 file.  Set to 8196 for now.
            else:
                if shape is None:
                    if channel_metadata is not None:
                        # can estimate a size, this will help with allocating
                        # and set the chunk size to a realistic value
                        if (
                            channel_metadata.time_period.start
                            != channel_metadata.time_period.end
                        ):
                            if channel_metadata.sample_rate > 0:
                                estimate_size = (
                                    int(
                                        (
                                            channel_metadata.time_period.end
                                            - channel_metadata.time_period.start
                                        )
                                        * channel_metadata.sample_rate
                                    ),
                                )
                        else:
                            estimate_size = (1,)
                            chunks = CHUNK_SIZE
                    else:
                        estimate_size = (1,)
                        chunks = CHUNK_SIZE
                    if estimate_size[0] > 2**31:
                        estimate_size = (1,)
                        self.logger.warning(
                            "Estimated size is too large. Check start and end "
                            "times, initializing with size (1,)"
                        )
                else:
                    estimate_size = shape
                ## Create the dataset
                channel_group = self.hdf5_group.create_dataset(
                    channel_name,
                    shape=estimate_size,
                    maxshape=max_shape,
                    dtype=channel_dtype,
                    chunks=chunks,
                    **self.dataset_options,
                )
            if channel_metadata:
                if channel_metadata.component != channel_name:
                    self.logger.warning(
                        f"Channel name {channel_name} != "
                        f"channel_metadata.component "
                        f"{channel_metadata.component}, setting to {channel_name}"
                    )
                    channel_metadata.component = channel_name
            if channel_type.lower() in ["magnetic"]:
                channel_obj = MagneticDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            elif channel_type.lower() in ["electric"]:
                channel_obj = ElectricDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            elif channel_type.lower() in ["auxiliary"]:
                channel_obj = AuxiliaryDataset(
                    channel_group, dataset_metadata=channel_metadata
                )
            else:
                msg = (
                    "`channel_type` must be in [ electric | magnetic | "
                    f"auxiliary ]. Input was {channel_type}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
        except (OSError, RuntimeError, ValueError):
            msg = f"channel {channel_name} already exists, returning existing group."
            self.logger.debug(msg)
            channel_obj = self.get_channel(channel_name)

            if data is not None:
                self.logger.debug(f"Replacing data with new shape {data.shape}")
                channel_obj.replace_dataset(data)

                self.logger.debug("Updating metadata")
                channel_obj.metadata.update(channel_metadata)
                channel_obj.write_metadata()
                self.logger.debug(f"Done with {channel_name}")
        # need to make sure the channel name is passed.
        if channel_obj.metadata.component != channel_name:
            channel_obj.metadata.component = channel_name
            channel_obj.write_metadata()

        # Cache the processed channel metadata to prevent duplicate processing in metadata property
        # Use the channel object's metadata which has already been processed through from_dict
        self._channel_metadata_cache[channel_name] = channel_obj.metadata

        return channel_obj

    def get_channel(
        self, channel_name: str
    ) -> ElectricDataset | MagneticDataset | AuxiliaryDataset | ChannelDataset:
        """
        Get a channel from an existing name.

        Returns the appropriate channel dataset container based on the
        channel type (electric, magnetic, or auxiliary).

        Parameters
        ----------
        channel_name : str
            Name of the channel to retrieve (e.g., 'ex', 'ey', 'hx').

        Returns
        -------
        ElectricDataset or MagneticDataset or AuxiliaryDataset or ChannelDataset
            Channel dataset object containing the channel data and metadata.

        Raises
        ------
        MTH5Error
            If the channel does not exist in the run.

        Examples
        --------
        Attempting to get a non-existent channel:

        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5("example.h5", mode='r')
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> ex = run.get_channel('ex')
        MTH5Error: ex does not exist, check groups_list for existing names

        Check available channels first:

        >>> run.groups_list
        ['ey', 'hx', 'hz']

        Get an existing channel:

        >>> ey = run.get_channel('ey')
        >>> print(ey)
        Channel Electric:
        -------------------
                component:        ey
                data type:        electric
                data format:      float32
                data shape:       (4096,)
                start:            1980-01-01T00:00:00+00:00
                end:              1980-01-01T00:00:01+00:00
                sample rate:      4096
        """

        channel_name = validate_name(channel_name.lower())
        try:
            ch_dataset = self.hdf5_group[channel_name]
        except KeyError:
            msg = (
                f"{channel_name} does not exist, check groups_list "
                "for existing names"
            )
            self.logger.debug(msg)
            raise MTH5Error(msg)
        if ch_dataset.attrs["mth5_type"].lower() in ["electric"]:
            channel = ElectricDataset(
                ch_dataset,
            )
        elif ch_dataset.attrs["mth5_type"].lower() in ["magnetic"]:
            channel = MagneticDataset(
                ch_dataset,
            )
        elif ch_dataset.attrs["mth5_type"].lower() in ["auxiliary"]:
            channel = AuxiliaryDataset(
                ch_dataset,
            )
        else:
            channel = ChannelDataset(ch_dataset)
        channel.read_metadata()

        return channel

    def remove_channel(self, channel_name: str) -> None:
        """
        Remove a channel from the run.

        Deleting a channel is not as simple as del(channel). In HDF5,
        this does not free up memory; it simply removes the reference
        to that channel. The common way to get around this is to
        copy what you want into a new file, or overwrite the channel.

        Parameters
        ----------
        channel_name : str
            Name of the existing channel to remove.

        Notes
        -----
        Deleting a channel does not reduce the HDF5 file size. It simply
        removes the reference. If file size reduction is your goal, copy
        what you want into another file.

        Todo: Need to remove summary table entry as well.

        Examples
        --------
        >>> from mth5 import mth5
        >>> mth5_obj = mth5.MTH5()
        >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
        >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
        >>> run.remove_channel('ex')
        """

        channel_name = validate_name(channel_name.lower())

        try:
            del self.hdf5_group[channel_name]
            # Remove from metadata cache if present
            if channel_name in self._channel_metadata_cache:
                del self._channel_metadata_cache[channel_name]
            self.logger.info(
                "Deleting a channel does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{channel_name} does not exist, "
                "check groups_list for existing names"
            )
            self.logger.debug("Error: " + msg)
            raise MTH5Error(msg)

    def has_data(self) -> bool:
        """
        Check if the run contains any non-empty, non-zero data.

        Verifies that all channels in the run have valid data (non-zero and
        non-empty arrays). Returns False if any channel lacks data.

        Returns
        -------
        bool
            True if all channels have data, False if any channel is empty
            or all zeros.

        Notes
        -----
        A channel is considered to have data if its has_data() method
        returns True, meaning it contains non-zero values.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> if run.has_data():
        ...     print("Run contains valid data")
        ...     runts = run.to_runts()
        """
        has_data_list = []
        has_data = True
        for channel in self.groups_list:
            if channel in ["summary"]:
                continue
            ch_obj = self.get_channel(channel)
            has_data_list.append(f"{ch_obj.metadata.component}: {ch_obj.has_data()}")
            if not ch_obj.has_data():
                has_data = False

        if not has_data:
            self.logger.info(", ".join(has_data_list))
        return has_data

    def to_runts(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> RunTS:
        """
        Convert run to a RunTS timeseries object.

        Combines all channels in the run into a RunTS object which handles
        multi-channel time series data with associated metadata.

        Parameters
        ----------
        start : str, optional
            Start time for time slice in ISO format (e.g., '2023-01-01T12:00:00').
            If None, uses entire channel data. Default is None.
        end : str, optional
            End time for time slice in ISO format. Only used if start is specified.
            Default is None.
        n_samples : int, optional
            Number of samples to extract from start. If both end and n_samples
            are specified, end takes precedence. Default is None.

        Returns
        -------
        RunTS
            RunTS object containing all channels with full run and station metadata.

        Notes
        -----
        - Includes run, station, and survey metadata in the output
        - Skips the 'summary' group which is not a channel
        - If start is specified, performs time slicing; otherwise returns full data

        Examples
        --------
        Convert entire run to RunTS:

        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> runts = run.to_runts()
        >>> print(runts.channels)
        ['ex', 'ey', 'hx', 'hy']

        Time slice the run:

        >>> runts = run.to_runts(start='2023-01-01T12:00:00',
        ...                       end='2023-01-01T13:00:00')
        >>> print(runts.ex.ts.shape)
        (1024,)
        """
        ch_list = []
        for channel in self.groups_list:
            if channel in ["summary"]:
                continue
            ch_obj = self.get_channel(channel)

            if start is not None:
                ts_obj = ch_obj.time_slice(start, end=end, n_samples=n_samples)
            else:
                ts_obj = ch_obj.to_channel_ts()
            ch_list.append(ts_obj)
        return RunTS(
            ch_list,
            run_metadata=self.metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )

    def from_runts(
        self, run_ts_obj: RunTS, **kwargs: Any
    ) -> list[ElectricDataset | MagneticDataset | AuxiliaryDataset]:
        """
        Create channel datasets from a RunTS timeseries object.

        Converts a RunTS object with multiple channels and metadata into
        HDF5 channel datasets and updates run metadata accordingly.

        Parameters
        ----------
        run_ts_obj : RunTS
            RunTS object containing multiple channels and metadata.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[ElectricDataset | MagneticDataset | AuxiliaryDataset]
            List of created channel dataset objects.

        Raises
        ------
        MTH5Error
            If input is not a RunTS object.

        Notes
        -----
        - Updates run metadata from input object
        - Validates station and run IDs match current context
        - Creates appropriate channel type based on channel metadata
        - Automatically registers recorded channels in run metadata

        Examples
        --------
        >>> from mt_timeseries import RunTS
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> runts = RunTS.from_file("timeseries_data.txt")
        >>> channels = run.from_runts(runts)
        >>> print(f"Created {len(channels)} channels")
        Created 4 channels
        """

        if not isinstance(run_ts_obj, RunTS):
            msg = f"Input must be a mt_timeseries.RunTS object not {type(run_ts_obj)}"
            self.logger.error(msg)
            raise MTH5Error(msg)
        self._metadata.update(run_ts_obj.run_metadata)

        channels = []

        for comp in run_ts_obj.channels:
            ch = getattr(run_ts_obj, comp)

            if ch.station_metadata.id is not None:
                if ch.station_metadata.id != self.station_metadata.id:
                    if ch.station_metadata.id not in ["0", None]:
                        self.logger.warning(
                            f"Channel station.id {ch.station_metadata.id} != "
                            f" group station.id {self.station_metadata.id}. "
                            f"Setting to ch.station_metadata.id to {self.station_metadata.id}"
                        )
                        ch.station_metadata.id = self.station_metadata.id
            if ch.run_metadata.id is not None:
                if ch.run_metadata.id != self.metadata.id:
                    if ch.run_metadata.id not in ["0", None]:
                        self.logger.warning(
                            f"Channel run.id {ch.run_metadata.id} != "
                            f" group run.id {self.metadata.id}. "
                            f"Setting to ch.run_metadata.id to {self.metadata.id}"
                        )
                        ch.run_metadata.id = self.metadata.id

            channels.append(self.from_channel_ts(ch))
        self.update_metadata()
        return channels

    def from_channel_ts(
        self, channel_ts_obj: ChannelTS
    ) -> ElectricDataset | MagneticDataset | AuxiliaryDataset:
        """
        Create a channel dataset from a ChannelTS timeseries object.

        Converts a single ChannelTS object with time series data and metadata
        into an HDF5 channel dataset. Handles filter registration and updates
        run metadata with channel information.

        Parameters
        ----------
        channel_ts_obj : ChannelTS
            ChannelTS object containing time series data and metadata.

        Returns
        -------
        ElectricDataset | MagneticDataset | AuxiliaryDataset
            Created channel dataset object.

        Raises
        ------
        MTH5Error
            If input is not a ChannelTS object.

        Notes
        -----
        - Registers filters from channel response if present
        - Validates and corrects station/run ID mismatches
        - Updates run metadata recorded channel lists
        - Automatically determines channel type from metadata

        Examples
        --------
        >>> from mt_timeseries import ChannelTS
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> channel = ChannelTS.from_file("ex_timeseries.txt")
        >>> ex = run.from_channel_ts(channel)
        >>> print(ex.metadata.component)
        ex
        """

        if not isinstance(channel_ts_obj, ChannelTS):
            msg = f"Input must be a mt_timeseries.ChannelTS object not {type(channel_ts_obj)}"
            self.logger.error(msg)
            raise MTH5Error(msg)
        ## Need to add in the filters
        if channel_ts_obj.channel_response.filters_list != []:
            from mth5.groups import FiltersGroup

            fg = FiltersGroup(self.hdf5_group.parent.parent.parent["Filters"])
            for ff in channel_ts_obj.channel_response.filters_list:
                fg.add_filter(ff)
        ch_obj = self.add_channel(
            channel_ts_obj.component,
            channel_ts_obj.channel_metadata.type,
            channel_ts_obj.ts,
            channel_metadata=channel_ts_obj.channel_metadata,
        )

        # need to update the channels recorded
        if channel_ts_obj.channel_metadata.type == "electric":
            if self.metadata.channels_recorded_electric is None:
                self.metadata.channels_recorded_electric = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_electric
            ):
                self.metadata.channels_recorded_electric.append(
                    channel_ts_obj.component
                )
        elif channel_ts_obj.channel_metadata.type == "magnetic":
            if self.metadata.channels_recorded_magnetic is None:
                self.metadata.channels_recorded_magnetic = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component not in self.metadata.channels_recorded_magnetic
            ):
                self.metadata.channels_recorded_magnetic.append(
                    channel_ts_obj.component
                )
        elif channel_ts_obj.channel_metadata.type == "auxiliary":
            if self.metadata.channels_recorded_auxiliary is None:
                self.metadata.channels_recorded_auxiliary = [channel_ts_obj.component]
            elif (
                channel_ts_obj.component
                not in self.metadata.channels_recorded_auxiliary
            ):
                self.metadata.channels_recorded_auxiliary.append(
                    channel_ts_obj.component
                )
        return ch_obj

    def update_run_metadata(self) -> None:
        """
        Update metadata and table entries (Deprecated).
        .. deprecated::
            Use update_metadata() instead.
        Raises
        ------
        DeprecationWarning
            Always raised to indicate this method should not be used.
        """

        raise DeprecationWarning(
            "'update_run_metadata' has been deprecated use 'update_metadata()'"
        )

    def update_metadata(self) -> None:
        """
        Update run metadata from all channels and persist to HDF5.

        Aggregates metadata from all channels including time period and
        sample rate, then writes updated metadata to HDF5 attributes.

        Raises
        ------
        Exception
            May raise exceptions if no channels exist (logs warning).

        Notes
        -----
        Updates:

        - Time period start from minimum of all channels
        - Time period end from maximum of all channels
        - Sample rate from first channel (assumes uniform across channels)

        Should be called after adding or removing channels to maintain
        consistency between channel and run metadata.

        Examples
        --------
        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> run.add_channel('ex', 'electric', data=ex_data)
        >>> run.add_channel('ey', 'electric', data=ey_data)
        >>> run.update_metadata()  # Updates time period and sample rate
        """
        channel_summary = self.channel_summary.copy()

        self._metadata.time_period.start = channel_summary.start.min().isoformat()
        self._metadata.time_period.end = channel_summary.end.max().isoformat()
        try:
            self._metadata.sample_rate = channel_summary.sample_rate.unique()[0]
        except IndexError:
            msg = "There maybe no channels associated with this run -- setting sample_rate to 0"
            self.logger.critical(msg)
            self._metadata.sample_rate = 0
        self.write_metadata()

    def plot(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> Any:
        """
        Create a matplotlib plot of all channels in the run.

        Generates a multi-panel plot showing all channels in the run using
        the RunTS plotting functionality.

        Parameters
        ----------
        start : str, optional
            Start time for time slice in ISO format. If None, plots entire
            channel data. Default is None.
        end : str, optional
            End time for time slice in ISO format. Only used if start is
            specified. Default is None.
        n_samples : int, optional
            Number of samples to extract from start. If both end and n_samples
            are specified, end takes precedence. Default is None.

        Returns
        -------
        Any
            Matplotlib figure or axes object (depends on RunTS.plot() implementation).

        Notes
        -----
        - Creates separate subplots for each channel type (electric, magnetic, auxiliary)
        - Time slice parameters work the same as to_runts()
        - Requires matplotlib to be installed

        Examples
        --------
        Plot entire run:

        >>> run = mth5_obj.get_run("MT001", "MT001a")
        >>> fig = run.plot()
        >>> fig.show()

        Plot time slice:

        >>> fig = run.plot(start='2023-01-01T12:00:00',
        ...                end='2023-01-01T13:00:00')
        """
        runts = self.to_runts(start=start, end=end, n_samples=n_samples)

        return runts.plot()
