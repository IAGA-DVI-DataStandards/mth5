# -*- coding: utf-8 -*-
"""
Fourier Coefficient group management for MTH5 format.

This module provides classes for organizing and managing Fourier Coefficient
data at multiple decimation levels, including utilities for data import/export
with different formats (numpy, xarray, pandas).

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

"""

from __future__ import annotations

from typing import Optional

import h5py
import mt_metadata.processing.fourier_coefficients as fc

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import xarray as xr

from mth5.groups import BaseGroup, FCChannelDataset
from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

# from mth5.groups import FCGroup


# =============================================================================
"""fc -> FCMasterGroup -> FCGroup -> DecimationLevelGroup -> ChannelGroup -> FCChannelDataset"""


class MasterFCGroup(BaseGroup):
    """
    Master container for all Fourier Coefficient estimations of time series data.

    This class manages multiple Fourier Coefficient processing runs, each containing
    different decimation levels. No metadata is required at the master level.

    Hierarchy
    ---------
    MasterFCGroup -> FCGroup (processing runs) -> FCDecimationGroup (decimation levels)
    -> FCChannelDataset (individual channels)

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for the master FC container.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Examples
    --------
    >>> import h5py
    >>> from mth5.groups.fourier_coefficients import MasterFCGroup
    >>> with h5py.File('data.h5', 'r') as f:
    ...     master = MasterFCGroup(f['FC'])
    ...     fc_group = master.add_fc_group('processing_run_1')
    """

    def __init__(self, group: h5py.Group, **kwargs) -> None:
        super().__init__(group, **kwargs)

    @property
    def fc_summary(self) -> pd.DataFrame:
        """
        Get a summary of all Fourier Coefficient processing runs.

        Returns
        -------
        pd.DataFrame
            Summary information for all FC groups including names and metadata.

        Examples
        --------
        >>> master = MasterFCGroup(h5_group)
        >>> summary = master.fc_summary
        """

    def add_fc_group(
        self,
        fc_name: str,
        fc_metadata: Optional[fc.Decimation] = None,
    ) -> FCGroup:
        """
        Add a Fourier Coefficient processing run group.

        Parameters
        ----------
        fc_name : str
            Name for the FC group (usually identifies the processing run).
        fc_metadata : fc.Decimation, optional
            Metadata for the FC group. Default is None.

        Returns
        -------
        FCGroup
            Newly created Fourier Coefficient group.

        Examples
        --------
        >>> master = MasterFCGroup(h5_group)
        >>> fc_group = master.add_fc_group('processing_run_1')
        >>> print(fc_group.name)
        'processing_run_1'
        """

        return self._add_group(fc_name, FCGroup, group_metadata=fc_metadata, match="id")

    def get_fc_group(self, fc_name: str) -> FCGroup:
        """
        Retrieve a Fourier Coefficient group by name.

        Parameters
        ----------
        fc_name : str
            Name of the FC group to retrieve.

        Returns
        -------
        FCGroup
            The requested Fourier Coefficient group.

        Raises
        ------
        MTH5Error
            If the FC group does not exist.

        Examples
        --------
        >>> master = MasterFCGroup(h5_group)
        >>> fc_group = master.get_fc_group('processing_run_1')
        """
        return self._get_group(fc_name, FCGroup)

    def remove_fc_group(self, fc_name: str) -> None:
        """
        Remove a Fourier Coefficient group.

        Deletes the specified FC group and all associated decimation levels and channels.

        Parameters
        ----------
        fc_name : str
            Name of the FC group to remove.

        Raises
        ------
        MTH5Error
            If the FC group does not exist.

        Examples
        --------
        >>> master = MasterFCGroup(h5_group)
        >>> master.remove_fc_group('processing_run_1')
        """

        self._remove_group(fc_name)


class FCDecimationGroup(BaseGroup):
    """
    Container for a single decimation level of Fourier Coefficient data.

    This class manages all channels at a specific decimation level, assuming
    uniform sampling in both frequency and time domains.

    Data Assumptions
    ----------------
    1. Data uniformly sampled in frequency domain
    2. Data uniformly sampled in time domain
    3. FFT moving window has uniform step size

    Attributes
    ----------
    start_time : datetime
        Start time of the decimation level
    end_time : datetime
        End time of the decimation level
    channels : list
        List of channel names in this decimation level
    decimation_factor : int
        Factor by which data was decimated
    decimation_level : int
        Level index in decimation hierarchy
    sample_rate : float
        Sample rate after decimation (Hz)
    method : str
        Method used (FFT, wavelet, etc.)
    window : dict
        Window parameters (length, overlap, type, sample rate)

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this decimation level.
    decimation_level_metadata : optional
        Metadata for the decimation level. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Examples
    --------
    >>> decimation = FCDecimationGroup(h5_group, decimation_level_metadata=metadata)
    >>> channel = decimation.add_channel('Ex', fc_data=fc_array)
    """

    def __init__(
        self,
        group: h5py.Group,
        decimation_level_metadata: Optional[fc.Decimation] = None,
        **kwargs,
    ) -> None:
        super().__init__(group, group_metadata=decimation_level_metadata, **kwargs)

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include channel information in the runs"""

        self._metadata.channels = []
        for ch in self.groups_list:
            ch_group = self.get_channel(ch)
            self._metadata.channels.append(ch_group.metadata)
            self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def channel_summary(self) -> pd.DataFrame:
        """
        Get a summary of all channels in this decimation level.

        Returns a pandas DataFrame with detailed information about each Fourier
        Coefficient channel including time ranges, dimensions, and sampling rates.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - component : str
                Channel component name (e.g., 'Ex', 'Hy')
            - start : datetime64[ns]
                Start time of the channel data
            - end : datetime64[ns]
                End time of the channel data
            - n_frequency : int64
                Number of frequency bins
            - n_windows : int64
                Number of time windows
            - sample_rate_decimation_level : float64
                Decimation level sample rate (Hz)
            - sample_rate_window_step : float64
                Sample rate of window stepping (Hz)
            - units : str
                Physical units of the data
            - hdf5_reference : h5py.ref_dtype
                HDF5 reference to the channel dataset

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> summary = decimation.channel_summary
        >>> print(summary[['component', 'n_frequency', 'n_windows']])
        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FCChannel"]:
                    ch_list.append(
                        (
                            group.attrs["component"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.shape[0],
                            group.shape[1],
                            group.attrs["sample_rate_decimation_level"],
                            group.attrs["sample_rate_window_step"],
                            group.attrs["units"],
                            group.ref,
                        )
                    )
            except KeyError as error:
                self.logger.debug(f"Cannot find a key: {error}")
        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("n_frequency", np.int64),
                    ("n_windows", np.int64),
                    ("sample_rate_decimation_level", np.float64),
                    ("sample_rate_window_step", np.float64),
                    ("units", "U25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def from_dataframe(
        self,
        df: pd.DataFrame,
        channel_key: str,
        time_key: str = "time",
        frequency_key: str = "frequency",
    ) -> None:
        """
        Load Fourier Coefficient data from a pandas DataFrame.

        Assumes the channel_key column contains complex coefficient values
        organized with time and frequency dimensions.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the coefficient data.
        channel_key : str
            Name of the column containing coefficient values.
        time_key : str, default='time'
            Name of the time coordinate column.
        frequency_key : str, default='frequency'
            Name of the frequency coordinate column.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> decimation.from_dataframe(df, channel_key='Ex', time_key='time')
        """

        if not isinstance(df, pd.DataFrame):
            msg = f"Must input a pandas dataframe not {type(df)}"
            self.logger.error(msg)
            raise TypeError(msg)
        for col in df.columns:
            df[col] = np.complex128(df[col])
            xrds = df[col].to_xarray()
            self.add_channel(col, fc_data=xrds.to_numpy())

    def from_xarray(
        self,
        data_array: xr.Dataset | xr.DataArray,
        sample_rate_decimation_level: float,
    ) -> None:
        """
        Load Fourier Coefficient data from an xarray DataArray or Dataset.

        Automatically extracts metadata (time, frequency, units) from the xarray
        object and creates appropriate FCChannelDataset instances for each
        variable or the single DataArray.

        Parameters
        ----------
        data_array : xr.DataArray or xr.Dataset
            Input xarray object with 'time' and 'frequency' coordinates and
            dimensions ['time', 'frequency'] (or transposed variant).
        sample_rate_decimation_level : float
            Sample rate of the decimation level (Hz).

        Raises
        ------
        TypeError
            If data_array is not an xarray Dataset or DataArray.

        Notes
        -----
        Automatically handles both (time, frequency) and (frequency, time)
        dimension ordering. Units are extracted from xarray attributes if available.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> decimation = FCDecimationGroup(h5_group)

        Create sample xarray data:

        >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
        >>> freqs = np.linspace(0.01, 100, 256)
        >>> data_array = np.random.randn(len(times), len(freqs)) + \\
        ...              1j * np.random.randn(len(times), len(freqs))
        >>> xr_data = xr.DataArray(
        ...     data_array,
        ...     dims=['time', 'frequency'],
        ...     coords={'time': times, 'frequency': freqs},
        ...     name='Ex'
        ... )

        Load into decimation group:

        >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)
        """

        if not isinstance(data_array, (xr.Dataset, xr.DataArray)):
            msg = f"Must input a xarray Dataset or DataArray not {type(data_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
        ch_metadata = fc.FCChannel()
        ch_metadata.time_period.start = data_array.time[0].values
        ch_metadata.time_period.end = data_array.time[-1].values
        ch_metadata.sample_rate_decimation_level = sample_rate_decimation_level
        ch_metadata.frequency_min = data_array.coords["frequency"].data.min()
        ch_metadata.frequency_max = data_array.coords["frequency"].data.max()
        step_size = (
            data_array.coords["time"].data[1] - data_array.coords["time"].data[0]
        )
        ch_metadata.sample_rate_window_step = step_size / np.timedelta64(1, "s")
        try:
            ch_metadata.units = data_array.units
        except AttributeError:
            self.logger.debug("Could not find 'units' in xarray")
        if isinstance(data_array, xr.DataArray):
            self._set_prewhitening_metadata(ch_metadata, data_array)
            self.add_channel(
                data_array.name,
                fc_data=data_array.to_numpy(),
                fc_metadata=ch_metadata,
            )
        else:
            for ch in data_array.data_vars.keys():
                ch_metadata.component = ch
                self._set_prewhitening_metadata(ch_metadata, data_array[ch])
                if ch in self.channel_summary.component.to_list():
                    self.remove_channel(ch)
                # time index should be the first index
                if data_array[ch].time.size == data_array[ch].shape[0]:
                    self.add_channel(
                        ch,
                        fc_data=data_array[ch].to_numpy(),
                        fc_metadata=ch_metadata,
                        dtype=data_array[ch].dtype,
                    )
                elif data_array[ch].time.size == data_array[ch].shape[1]:
                    self.add_channel(
                        ch,
                        fc_data=data_array[ch].to_numpy().T,
                        fc_metadata=ch_metadata,
                        dtype=data_array[ch].dtype,
                    )
        return

    @staticmethod
    def _set_prewhitening_metadata(
        channel_metadata: fc.FCChannel,
        data_array: xr.DataArray,
    ) -> None:
        """Copy fitted prewhitening metadata from a Fourier data array."""
        result = data_array.attrs.get("prewhitening")
        if result is None:
            return

        channel_metadata.prewhitening_type = data_array.attrs["prewhitening_type"]
        channel_metadata.prewhitening_fit_domain = data_array.attrs[
            "prewhitening_fit_domain"
        ]
        channel_metadata.prewhitening_response_convention = data_array.attrs[
            "prewhitening_response_convention"
        ]
        channel_metadata.prewhitening_ar_coefficients = result["ar_coefficients"]
        channel_metadata.prewhitening_ma_coefficients = result["ma_coefficients"]

    def to_xarray(self, channels: Optional[list[str]] = None) -> xr.Dataset:
        """
        Create an xarray Dataset from Fourier Coefficient channels.

        If no channels are specified, all channels in the decimation level
        are included. Each channel becomes a data variable in the resulting Dataset.

        Parameters
        ----------
        channels : list[str], optional
            List of channel names to include. If None, all channels are used.
            Default is None.

        Returns
        -------
        xr.Dataset
            xarray Dataset with channels as data variables and 'time' and
            'frequency' as shared coordinates.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> xr_data = decimation.to_xarray()
        >>> print(xr_data.data_vars)
        Data variables:
            Ex  (time, frequency) complex128
            Ey  (time, frequency) complex128

        Get specific channels:

        >>> subset = decimation.to_xarray(channels=['Ex', 'Ey'])
        """

        if channels is None:
            channels = self.groups_list
        ch_dict = {}
        for ch in channels:
            ch_ds = self.get_channel(ch)
            ch_dict[ch] = ch_ds.to_xarray()
        return xr.Dataset(ch_dict)

    def from_numpy_array(
        self,
        nd_array: np.ndarray,
        ch_name: str | list[str],
    ) -> None:
        """
        Load Fourier Coefficient data from a numpy array.

        Assumes array shape is either (n_frequencies, n_windows) for a single
        channel or (n_channels, n_frequencies, n_windows) for multiple channels.

        Parameters
        ----------
        nd_array : np.ndarray
            Input numpy array containing coefficient data.
        ch_name : str or list[str]
            Channel name (for 2D array) or list of channel names
            (for 3D array).

        Raises
        ------
        TypeError
            If nd_array is not a numpy ndarray.
        ValueError
            If array shape is not (n_frequencies, n_windows) or
            (n_channels, n_frequencies, n_windows).

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)

        Load single channel:

        >>> data_2d = np.random.randn(256, 100) + 1j * np.random.randn(256, 100)
        >>> decimation.from_numpy_array(data_2d, ch_name='Ex')

        Load multiple channels:

        >>> data_3d = np.random.randn(2, 256, 100) + 1j * np.random.randn(2, 256, 100)
        >>> decimation.from_numpy_array(data_3d, ch_name=['Ex', 'Ey'])
        """

        if not isinstance(nd_array, np.ndarray):
            msg = f"Must input a numpy ndarray not {type(nd_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
        if len(nd_array.shape) == 3:
            for index, ch in zip(nd_array.shape[0], ch_name):
                self.add_channel(ch, fc_data=nd_array[index])
        elif len(nd_array.shape) == 2:
            self.add_channel(ch_name, fc_data=nd_array)
        else:
            raise ValueError(
                "input array must be shaped (n_frequencies, n_windows) or "
                "(n_channels, n_frequencies, n_windows)"
            )

    def add_channel(
        self,
        fc_name: str,
        fc_data: Optional[np.ndarray] = None,
        fc_metadata: Optional[fc.FCChannel] = None,
        max_shape: tuple = (None, None),
        chunks: bool = True,
        dtype: type = complex,
        **kwargs,
    ) -> FCChannelDataset:
        """
        Add a Fourier Coefficient channel to the decimation level.

        Creates a new FCChannelDataset for a single channel at a single
        decimation level. Input data can be provided as numpy array or created empty.

        Parameters
        ----------
        fc_name : str
            Name for the Fourier Coefficient channel (usually component name like 'Ex').
        fc_data : np.ndarray, optional
            Input data with shape (n_frequencies, n_windows). Default is None (creates empty).
        fc_metadata : fc.FCChannel, optional
            Metadata for the channel. Default is None.
        max_shape : tuple, default=(None, None)
            Maximum shape for HDF5 dataset dimensions (expandable if None).
        chunks : bool, default=True
            Whether to use HDF5 chunking.
        dtype : type, default=complex
            Data type for the dataset.
        **kwargs
            Additional keyword arguments for HDF5 dataset creation.

        Returns
        -------
        FCChannelDataset
            Newly created FCChannelDataset object.

        Raises
        ------
        TypeError
            If fc_data type is not supported.

        Notes
        -----
        Data layout assumes (time, frequency) organization:

        - time index: window start times
        - frequency index: harmonic indices or float values
        - data: complex Fourier coefficients

        If a channel with the same name already exists, the existing channel
        is returned instead of creating a duplicate.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> metadata = fc.FCChannel(component='Ex')

        Create from numpy array:

        >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
        >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

        Create empty channel (expandable):

        >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)
        """

        fc_name = validate_name(fc_name)

        if fc_metadata is None:
            fc_metadata = fc.FCChannel(name=fc_name)
        if fc_data is not None:
            if not isinstance(
                fc_data, (np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame)
            ):
                msg = (
                    "Need to input a numpy.array, xarray.DataArray, "
                    f"xr.Dataset, pd.DataFrame not {type(fc_data)}"
                )
                self.logger.exception(msg)
                raise TypeError(msg)
        else:
            chunks = True
            fc_data = np.zeros((1, 1), dtype=dtype)
        try:
            dataset = self.hdf5_group.create_dataset(
                fc_name,
                data=fc_data,
                dtype=dtype,
                chunks=chunks,
                maxshape=max_shape,
                **self.dataset_options,
            )

            fc_dataset = FCChannelDataset(dataset, dataset_metadata=fc_metadata)
        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
            msg = f"estimate {fc_metadata.component} already exists, returning existing group."
            self.logger.debug(msg)

            fc_dataset = self.get_channel(fc_metadata.component)
        return fc_dataset

    def get_channel(self, fc_name: str) -> FCChannelDataset:
        """
        Retrieve a Fourier Coefficient channel by name.

        Parameters
        ----------
        fc_name : str
            Name of the Fourier Coefficient channel to retrieve.

        Returns
        -------
        FCChannelDataset
            The requested Fourier Coefficient channel dataset.

        Raises
        ------
        KeyError
            If the channel does not exist in this decimation level.
        MTH5Error
            If unable to retrieve the channel from HDF5.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> channel = decimation.get_channel('Ex')
        >>> print(channel.shape)
        (100, 256)
        """
        fc_name = validate_name(fc_name)

        try:
            fc_dataset = self.hdf5_group[fc_name]
            fc_metadata = fc.FCChannel(**dict(fc_dataset.attrs))
            return FCChannelDataset(fc_dataset, dataset_metadata=fc_metadata)
        except KeyError:
            msg = f"{fc_name} does not exist, check groups_list for existing names"
            self.logger.error(msg)
            raise MTH5Error(msg)
        except OSError as error:
            self.logger.error(error)
            raise MTH5Error(error)

    def remove_channel(self, fc_name: str) -> None:
        """
        Remove a Fourier Coefficient channel from the decimation level.

        Deletes the HDF5 dataset associated with the channel. Note that this
        removes the reference but does not reduce the HDF5 file size.

        Parameters
        ----------
        fc_name : str
            Name of the Fourier Coefficient channel to remove.

        Raises
        ------
        MTH5Error
            If the channel does not exist.

        Notes
        -----
        Deleting a channel does not reduce the HDF5 file size; it simply
        removes the reference to the data. To truly reduce file size, copy
        the desired data to a new file.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> decimation.remove_channel('Ex')
        """
        fc_name = validate_name(fc_name.lower())

        try:
            del self.hdf5_group[fc_name]
            self.logger.info(
                "Deleting a estimate does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = f"{fc_name} does not exist, check groups_list for existing names"
            self.logger.error(msg)
            raise MTH5Error(msg)

    def update_metadata(self) -> None:
        """
        Update decimation level metadata from all channels.

        Aggregates metadata from all FC channels in the decimation level
        including time period, sample rates, and window step information.
        Updates the internal metadata object and writes to HDF5.

        Notes
        -----
        Collects the following information from channels:

        - Time period start/end from channel data
        - Sample rate decimation level
        - Sample rate window step

        Should be called after adding or modifying channels to keep
        metadata synchronized.

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> decimation.add_channel('Ex', fc_data=data_ex)
        >>> decimation.add_channel('Ey', fc_data=data_ey)
        >>> decimation.update_metadata()
        """
        channel_summary = self.channel_summary.copy()

        if not channel_summary.empty:
            self._metadata.time_period.start = channel_summary.start.min().isoformat()
            self._metadata.time_period.end = channel_summary.end.max().isoformat()
            self._metadata.sample_rate_decimation_level = (
                channel_summary.sample_rate_decimation_level.unique()[0]
            )
            self._metadata.sample_rate_window_step = (
                channel_summary.sample_rate_window_step.unique()[0]
            )
            self.write_metadata()

    def add_feature(
        self,
        feature_name: str,
        feature_data: Optional[np.ndarray] = None,
        feature_metadata: Optional[dict] = None,
        max_shape: tuple = (None, None, None),
        chunks: bool = True,
        **kwargs,
    ) -> None:
        """
        Add a feature dataset to the decimation level.

        Creates a new dataset for auxiliary features or derived quantities
        related to Fourier Coefficients (e.g., SNR, coherency, power, etc.).

        Parameters
        ----------
        feature_name : str
            Name for the feature dataset.
        feature_data : np.ndarray, optional
            Input data for the feature. Default is None (creates empty).
        feature_metadata : dict, optional
            Metadata dictionary for the feature. Default is None.
        max_shape : tuple, default=(None, None, None)
            Maximum shape for HDF5 dataset dimensions (expandable if None).
        chunks : bool, default=True
            Whether to use HDF5 chunking.
        **kwargs
            Additional keyword arguments for HDF5 dataset creation.

        Notes
        -----
        Feature types may include:

        - Power: Total power in Fourier coefficients
        - SNR: Signal-to-noise ratio
        - Coherency: Cross-component coherence
        - Weights: Channel-specific weights
        - Flags: Data quality or processing flags

        Examples
        --------
        >>> decimation = FCDecimationGroup(h5_group)
        >>> snr_data = np.random.randn(100, 256)
        >>> decimation.add_feature('snr', feature_data=snr_data)

        Or create empty feature for later population:

        >>> decimation.add_feature('power_Ex')
        """


class FCGroup(BaseGroup):
    """
    Manage a set of Fourier Coefficients from a single processing run.

    Holds Fourier Coefficient estimations organized by decimation level.
    Each decimation level contains channels (Ex, Ey, Hz, etc.) with complex
    frequency or time-frequency representations of the input signal.

    All channels must use the same calibration. Recalibration requires
    rerunning the Fourier Coefficient estimation.

    Attributes
    ----------
    hdf5_group : h5py.Group
        The HDF5 group containing decimation levels
    metadata : fc.Decimation
        Decimation metadata including time period, sample rates, and channels

    Notes
    -----
    Processing run structure:

    - Multiple decimation levels at different sample rates
    - Each decimation level contains multiple channels
    - Each channel contains complex Fourier coefficients
    - Time period and sample rates define the estimation window

    Examples
    --------
    >>> with h5py.File('data.h5', 'r') as f:
    ...     fc_run = FCGroup(f['Fourier_Coefficients/run_1'])
    ...     print(fc_run.decimation_level_summary)
    """

    def __init__(
        self,
        group: h5py.Group,
        decimation_level_metadata: Optional[fc.Decimation] = None,
        **kwargs,
    ) -> None:
        """
        Initialize FCGroup.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group containing decimation levels.
        decimation_level_metadata : fc.Decimation, optional
            Metadata object for the processing run. Default is None.
        **kwargs
            Additional keyword arguments passed to BaseGroup.
        """
        super().__init__(group, group_metadata=decimation_level_metadata, **kwargs)

    @BaseGroup.metadata.getter
    def metadata(self) -> fc.Decimation:
        """
        Get processing run metadata including all decimation levels.

        Collects metadata from all decimation level groups and aggregates
        into a single Decimation metadata object.

        Returns
        -------
        fc.Decimation
            Metadata containing time period, sample rates, and all decimation
            level information.

        Notes
        -----
        This getter automatically populates:

        - Time period (start and end)
        - List of all decimation levels and their metadata
        - HDF5 reference to this group

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> metadata = fc_run.metadata
        >>> print(metadata.time_period.start)
        2023-01-01T00:00:00
        """
        self._metadata.channels = []
        for dl in self.groups_list:
            dl_group = self.get_decimation_level(dl)
            self._metadata.levels.append(dl_group.metadata)
        self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def decimation_level_summary(self) -> pd.DataFrame:
        """
        Get a summary of all decimation levels in this processing run.

        Returns information about each decimation level including sample rate,
        decimation level value, and time span.

        Returns
        -------
        pd.DataFrame
            Summary with columns:

            - decimation_level: Integer decimation level identifier
            - start: ISO format start time of this decimation level
            - end: ISO format end time of this decimation level
            - hdf5_reference: Reference to the HDF5 group

        Notes
        -----
        Each row represents a single decimation level containing multiple
        channels with Fourier coefficients at different sample rates.

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> summary = fc_run.decimation_level_summary
        >>> print(summary[['decimation_level', 'start', 'end']])
           decimation_level                start                  end
        0              0     2023-01-01T00:00:00.000000  2023-01-01T01:00:00.000000
        1              1     2023-01-01T00:00:00.000000  2023-01-01T02:00:00.000000
        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FCDecimation"]:
                    ch_list.append(
                        (
                            group.attrs["decimation_level"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.ref,
                        )
                    )
            except KeyError as error:
                self.logger.debug(f"Could not find key: {error}")

        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def add_decimation_level(
        self,
        decimation_level_name: str,
        decimation_level_metadata: Optional[dict | fc.Decimation] = None,
    ) -> FCDecimationGroup:
        """
        Add a new decimation level to the processing run.

        Creates a new FCDecimationGroup for a single decimation level containing
        Fourier Coefficient channels at a specific sample rate.

        Parameters
        ----------
        decimation_level_name : str
            Identifier for the decimation level.
        decimation_level_metadata : dict | fc.Decimation, optional
            Metadata for the decimation level. Can be a dictionary or
            fc.Decimation object. Default is None.

        Returns
        -------
        FCDecimationGroup
            Newly created decimation level group.

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> metadata = fc.Decimation(decimation_level=0)
        >>> decimation = fc_run.add_decimation_level('0', metadata)
        """

        return self._add_group(
            decimation_level_name,
            FCDecimationGroup,
            group_metadata=decimation_level_metadata,
            match="decimation_level",
        )

    def get_decimation_level(self, decimation_level_name: str) -> FCDecimationGroup:
        """
        Retrieve a decimation level by name.

        Parameters
        ----------
        decimation_level_name : str
            Name or identifier of the decimation level.

        Returns
        -------
        FCDecimationGroup
            The requested decimation level group.

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> decimation = fc_run.get_decimation_level('0')
        >>> channels = decimation.groups_list
        """
        return self._get_group(decimation_level_name, FCDecimationGroup)

    def remove_decimation_level(self, decimation_level_name: str) -> None:
        """
        Remove a decimation level from the processing run.

        Deletes the HDF5 group and all its channels (FCChannelDataset objects).

        Parameters
        ----------
        decimation_level_name : str
            Name or identifier of the decimation level to remove.

        Notes
        -----
        This removes the entire decimation level and all channels within it.
        To remove individual channels, use FCDecimationGroup.remove_channel()
        instead.

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> fc_run.remove_decimation_level('0')
        """

        self._remove_group(decimation_level_name)

    def update_metadata(self) -> None:
        """
        Update processing run metadata from all decimation levels.

        Aggregates time period information from all decimation levels
        and writes updated metadata to HDF5.

        Notes
        -----
        Collects:

        - Earliest start time across all decimation levels
        - Latest end time across all decimation levels

        Should be called after adding or removing decimation levels.

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> fc_run.add_decimation_level('0', metadata0)
        >>> fc_run.add_decimation_level('1', metadata1)
        >>> fc_run.update_metadata()
        """
        decimation_level_summary = self.decimation_level_summary.copy()
        if not decimation_level_summary.empty:
            self._metadata.time_period.start = (
                decimation_level_summary.start.min().isoformat()
            )
            self._metadata.time_period.end = (
                decimation_level_summary.end.max().isoformat()
            )
            self.write_metadata()

    def supports_aurora_processing_config(
        self,
        processing_config: "aurora.config.metadata.processing.Processing",
        remote: bool,
    ) -> bool:
        """
        Check if all required decimation levels exist for Aurora processing.

        Performs an all-or-nothing check: returns True only if every decimation
        level required by the processing config is available in this FCGroup.

        Uses sequential logic to short-circuit: if any required decimation level
        is missing, immediately returns False without checking remaining levels.

        Parameters
        ----------
        processing_config : aurora.config.metadata.processing.Processing
            Aurora processing configuration containing required decimation levels.
        remote : bool
            Whether to check for remote processing compatibility.

        Returns
        -------
        bool
            True if all required decimation levels are available and consistent,
            False otherwise.

        Notes
        -----
        Validation logic:

        1. Extract list of decimation levels from processing config
        2. Iterate through each required level in sequence
        3. For each level, find a matching FCDecimation in this group
        4. Check consistency using Aurora's validation method
        5. If any level is missing or inconsistent, return False immediately
        6. Return True only if all levels pass validation

        Examples
        --------
        >>> fc_run = FCGroup(h5_group)
        >>> config = aurora.config.metadata.processing.Processing(...)
        >>> if fc_run.supports_aurora_processing_config(config, remote=False):
        ...     # All decimation levels are available
        ...     pass
        """
        pre_existing_fc_decimation_ids_to_check = self.groups_list
        levels_present = np.full(processing_config.num_decimation_levels, False)

        for i, aurora_decimation_level in enumerate(processing_config.decimations):
            # Quit checking if dec_level wasn't there
            if i > 0:
                if not levels_present[i - 1]:
                    return False

            # iterate over existing decimations
            for fc_decimation_id in pre_existing_fc_decimation_ids_to_check:
                fc_dec_group = self.get_decimation_level(fc_decimation_id)
                fc_decimation = fc_dec_group.metadata
                levels_present[i] = (
                    aurora_decimation_level.is_consistent_with_archived_fc_parameters(
                        fc_decimation=fc_decimation, remote=remote
                    )
                )
                if levels_present[i]:
                    pre_existing_fc_decimation_ids_to_check.remove(
                        fc_decimation_id
                    )  # no need to check this one again
                    break  # break inner for-loop over decimations

        return levels_present.all()
