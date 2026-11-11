# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:40:34 2024

@author: jpeacock
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from mt_metadata.features import FeatureDecimationChannel
from mt_metadata.processing.fourier_coefficients.decimation import Decimation

from mth5.groups import BaseGroup, FeatureChannelDataset, RunGroup
from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

# =============================================================================
"""feature -> FeatureMasterGroup -> FeatureGroup -> DecimationLevelGroup -> ChannelGroup -> FeatureChannelDataset"""

TIME_DOMAIN = ["ts", "time", "time series", "time_series"]
FREQUENCY_DOMAIN = ["fc", "frequency", "fourier", "fourier_domain"]


class MasterFeaturesGroup(BaseGroup):
    """
    Master group container for features associated with Fourier Coefficients or time series.

    This class manages the top-level organization of geophysical feature data,
    organizing it into feature-specific groups. Features can include various
    frequency or time-domain analyses.

    Hierarchy
    ---------
    MasterFeatureGroup -> FeatureGroup -> FeatureRunGroup ->

    - FC: FeatureDecimationGroup -> FeatureChannelDataset
    - Time Series: FeatureChannelDataset

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this MasterFeaturesGroup.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Examples
    --------
    >>> import h5py
    >>> from mth5.groups.features import MasterFeaturesGroup
    >>> with h5py.File('data.h5', 'r') as f:
    ...     master = MasterFeaturesGroup(f['features'])
    ...     feature_list = master.groups_list
    """

    def __init__(self, group: h5py.Group, **kwargs) -> None:
        super().__init__(group, **kwargs)

    def add_feature_group(
        self,
        feature_name: str,
        feature_metadata: Optional[FeatureDecimationChannel] = None,
    ) -> FeatureGroup:
        """
        Add a feature group to the master features container.

        Creates a new FeatureGroup with the specified name and optional metadata.
        Feature groups organize all runs and decimation levels for a particular feature.

        Parameters
        ----------
        feature_name : str
            Name for the feature group. Will be validated and formatted.
        feature_metadata : FeatureDecimationChannel, optional
            Metadata describing the feature. Default is None.

        Returns
        -------
        FeatureGroup
            Newly created feature group object.

        Examples
        --------
        >>> master = MasterFeaturesGroup(h5_group)
        >>> feature = master.add_feature_group('coherency')
        >>> print(feature.name)
        'coherency'
        """

        return self._add_group(
            feature_name,
            FeatureGroup,
            group_metadata=feature_metadata,
            match="name",
        )

    def get_feature_group(self, feature_name: str) -> FeatureGroup:
        """
        Retrieve a feature group by name.

        Parameters
        ----------
        feature_name : str
            Name of the feature group to retrieve.

        Returns
        -------
        FeatureGroup
            The requested feature group.

        Raises
        ------
        MTH5Error
            If the feature group does not exist.

        Examples
        --------
        >>> master = MasterFeaturesGroup(h5_group)
        >>> feature = master.get_feature_group('coherency')
        >>> print(feature.name)
        'coherency'
        """
        return self._get_group(feature_name, FeatureGroup)

    def remove_feature_group(self, feature_name: str) -> None:
        """
        Remove a feature group from the master container.

        Deletes the specified feature group and its associated data from the
        HDF5 file. Note that this operation removes the reference but does not
        reduce the file size; copy desired data to a new file for size reduction.

        Parameters
        ----------
        feature_name : str
            Name of the feature group to remove.

        Raises
        ------
        MTH5Error
            If the feature group does not exist.

        Examples
        --------
        >>> master = MasterFeaturesGroup(h5_group)
        >>> master.remove_feature_group('coherency')
        """

        self._remove_group(feature_name)


class FeatureGroup(BaseGroup):
    """
    Container for a single feature set with all associated runs and decimation levels.

    This class manages feature-specific data including all processing runs and
    decimation levels. Features can include both Fourier Coefficient and time series data.

    Hierarchy
    ---------
    FeatureGroup -> FeatureRunGroup ->

    - FC: FeatureDecimationLevel -> FeatureChannelDataset
    - TS: FeatureChannelDataset

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this FeatureGroup.
    feature_metadata : optional
        Metadata specific to this feature. Should include description and parameters.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Notes
    -----
    Feature metadata should be specific to the feature and include descriptions
    of the feature and any parameters used in its computation.

    Examples
    --------
    >>> feature = FeatureGroup(h5_group, feature_metadata=metadata)
    >>> run_group = feature.add_feature_run_group('run_1', domain='fc')
    """

    def __init__(
        self,
        group: h5py.Group,
        feature_metadata: Optional[object] = None,
        **kwargs,
    ) -> None:
        super().__init__(group, group_metadata=feature_metadata, **kwargs)

    def add_feature_run_group(
        self,
        feature_name: str,
        feature_run_metadata: Optional[object] = None,
        domain: str = "fc",
    ) -> object:
        """
        Add a feature run group for a single feature.

        Creates either a Fourier Coefficient run group or a time series run group
        based on the specified domain. The domain can be determined from the metadata
        or explicitly provided.

        Parameters
        ----------
        feature_name : str
            Name for the feature run group.
        feature_run_metadata : optional
            Metadata for the feature run. If provided, domain is extracted from
            metadata.domain attribute. Default is None.
        domain : str, default='fc'
            Domain type for the data. Must be one of:

            - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
            - 'ts', 'time', 'time series', 'time_series': Time series

        Returns
        -------
        FeatureFCRunGroup or FeatureTSRunGroup
            Newly created feature run group.

        Raises
        ------
        ValueError
            If domain is not recognized.
        AttributeError
            If metadata does not have a domain attribute when metadata is provided.

        Examples
        --------
        >>> feature = FeatureGroup(h5_group)
        >>> fc_run = feature.add_feature_run_group('processing_run_1', domain='fc')
        >>> ts_run = feature.add_feature_run_group('ts_analysis', domain='ts')
        """
        if feature_run_metadata is not None:
            try:
                domain = feature_run_metadata.domain
            except AttributeError:
                raise AttributeError(
                    "Could not find attribute 'domain' in metadata object"
                )

        if domain in FREQUENCY_DOMAIN:
            return self._add_group(
                feature_name,
                FeatureFCRunGroup,
                group_metadata=feature_run_metadata,
                match="id",
            )
        elif domain in TIME_DOMAIN:
            return self._add_group(
                feature_name,
                FeatureTSRunGroup,
                group_metadata=feature_run_metadata,
                match="id",
            )
        else:
            raise ValueError(
                f"feature_type {domain} not supported. Use either 'fc' "
                "for Fourier Coefficent or 'ts' for time series."
            )

    def get_feature_run_group(
        self,
        feature_name: str,
        domain: str = "frequency",
    ) -> object:
        """
        Retrieve a feature run group by name and domain type.

        Parameters
        ----------
        feature_name : str
            Name of the feature run group to retrieve.
        domain : str, default='frequency'
            Domain type. Must be one of:

            - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
            - 'ts', 'time', 'time series', 'time_series': Time series

        Returns
        -------
        FeatureFCRunGroup or FeatureTSRunGroup
            The requested feature run group.

        Raises
        ------
        ValueError
            If domain is not recognized.
        MTH5Error
            If the feature run group does not exist.

        Examples
        --------
        >>> feature = FeatureGroup(h5_group)
        >>> fc_run = feature.get_feature_run_group('processing_run_1', domain='fc')
        """
        if domain in FREQUENCY_DOMAIN:
            return self._get_group(feature_name, FeatureFCRunGroup)
        elif domain in TIME_DOMAIN:
            return self._get_group(feature_name, FeatureTSRunGroup)
        else:
            raise ValueError(
                f"feature_type {domain} not supported. Use either 'fc' "
                "for Fourier Coefficent or 'ts' for time series."
            )

    def remove_feature_run_group(self, feature_name: str) -> None:
        """
        Remove a feature run group.

        Deletes the specified feature run group and all its associated data.
        Note that deletion removes the reference but does not reduce HDF5 file size.

        Parameters
        ----------
        feature_name : str
            Name of the feature run group to remove.

        Raises
        ------
        MTH5Error
            If the feature run group does not exist.

        Examples
        --------
        >>> feature = FeatureGroup(h5_group)
        >>> feature.remove_feature_run_group('processing_run_1')
        """

        self._remove_group(feature_name)


class FeatureTSRunGroup(BaseGroup):
    """
    Container for time series features from a processing or analysis run.

    This class wraps a RunGroup to manage time series data features while
    maintaining compatibility with the feature hierarchy structure.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this FeatureTSRunGroup.
    feature_run_metadata : optional
        Metadata for the feature run (same type as timeseries.Run).
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Notes
    -----
    This class uses methods from RunGroup for channel management, which may
    have performance implications due to multiple RunGroup instantiations.

    Examples
    --------
    >>> ts_run = FeatureTSRunGroup(h5_group, feature_run_metadata=metadata)
    >>> channel = ts_run.add_feature_channel('Ex', 'electric', data)
    """

    def __init__(
        self,
        group: h5py.Group,
        feature_run_metadata: Optional[object] = None,
        **kwargs,
    ) -> None:
        super().__init__(group, group_metadata=feature_run_metadata, **kwargs)

        ### Use methods from RunGroup (might be slow cause initiating multiple
        ### RunGroups)?
        self._run_group = RunGroup(group, feature_run_metadata=None)

    def add_feature_channel(
        self,
        channel_name: str,
        channel_type: str,
        data: Optional[np.ndarray] = None,
        channel_dtype: str = "int32",
        shape: Optional[tuple] = None,
        max_shape: tuple = (None,),
        chunks: bool = True,
        channel_metadata: Optional[object] = None,
        **kwargs,
    ) -> object:
        """
        Add a time series channel to the feature run group.

        Creates a new channel for time series data with the specified properties
        and optional metadata. Channel metadata should be a timeseries.Channel object.

        Parameters
        ----------
        channel_name : str
            Name for the channel.
        channel_type : str
            Type of channel (e.g., 'electric', 'magnetic').
        data : np.ndarray, optional
            Initial data for the channel. Default is None.
        channel_dtype : str, default='int32'
            Data type for the channel.
        shape : tuple, optional
            Shape of the channel data. Default is None.
        max_shape : tuple, default=(None,)
            Maximum shape for expandable dimensions.
        chunks : bool, default=True
            Whether to use chunking for the dataset.
        channel_metadata : optional
            Metadata object (timeseries.Channel type). Default is None.
        **kwargs
            Additional keyword arguments for dataset creation.

        Returns
        -------
        object
            Channel object from RunGroup.

        Examples
        --------
        >>> ts_run = FeatureTSRunGroup(h5_group)
        >>> channel = ts_run.add_feature_channel(
        ...     'Ex', 'electric', data=np.arange(1000))
        """

        return self._run_group.add_channel(
            channel_name,
            channel_type,
            data,
            channel_dtype=channel_dtype,
            shape=shape,
            max_shape=max_shape,
            chunks=chunks,
            channel_metadata=channel_metadata,
            **kwargs,
        )

    def get_feature_channel(self, channel_name: str) -> object:
        """
        Retrieve a feature channel by name.

        Parameters
        ----------
        channel_name : str
            Name of the channel to retrieve.

        Returns
        -------
        object
            Channel object from RunGroup.

        Raises
        ------
        MTH5Error
            If the channel does not exist.

        Examples
        --------
        >>> ts_run = FeatureTSRunGroup(h5_group)
        >>> channel = ts_run.get_feature_channel('Ex')
        """

        return self._run_group.get_channel(channel_name)

    def remove_feature_channel(self, channel_name: str) -> None:
        """
        Remove a feature channel from the run group.

        Parameters
        ----------
        channel_name : str
            Name of the channel to remove.

        Raises
        ------
        MTH5Error
            If the channel does not exist.

        Examples
        --------
        >>> ts_run = FeatureTSRunGroup(h5_group)
        >>> ts_run.remove_feature_channel('Ex')
        """
        self._run_group.remove_channel(channel_name)


class FeatureFCRunGroup(BaseGroup):
    """
    Container for Fourier Coefficient features from a processing run.

    This class manages Fourier Coefficient data organized by decimation levels,
    each containing multiple frequency channels with time-frequency data.

    Hierarchy
    ---------
    FeatureFCRunGroup -> FeatureDecimationGroup -> FeatureChannelDataset

    Attributes
    ----------
    metadata : Decimation
        Metadata including:

        - list of decimation levels
        - start time (earliest)
        - end time (latest)
        - method (fft, wavelet, ...)
        - list of channels used
        - starting sample rate
        - bands used
        - type (TS or FC)

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this FeatureFCRunGroup.
    feature_run_metadata : optional
        Decimation metadata for the feature run. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Examples
    --------
    >>> fc_run = FeatureFCRunGroup(h5_group, feature_run_metadata=metadata)
    >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)
    """

    def __init__(
        self,
        group: h5py.Group,
        feature_run_metadata: Optional[Decimation] = None,
        **kwargs,
    ) -> None:
        super().__init__(group, group_metadata=feature_run_metadata, **kwargs)

    @BaseGroup.metadata.getter
    def metadata(self) -> Decimation:
        """Overwrite get metadata to include channel information in the runs"""

        # self._metadata.channels = []
        # for dl in self.groups_list:
        #     dl_group = self.get_decimation_level(dl)
        #     self._metadata.levels.append(dl_group.metadata)
        self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def decimation_level_summary(self) -> pd.DataFrame:
        """
        Get a summary of all decimation levels in the run.

        Returns a pandas DataFrame with information about each decimation level
        including decimation factor, time range, and HDF5 reference.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - name : str
                Decimation level name
            - start : datetime64[ns]
                Start time of the decimation level
            - end : datetime64[ns]
                End time of the decimation level
            - hdf5_reference : h5py.ref_dtype
                HDF5 reference to the decimation level group

        Examples
        --------
        >>> fc_run = FeatureFCRunGroup(h5_group)
        >>> summary = fc_run.decimation_level_summary
        >>> print(summary[['name', 'start', 'end']])
        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FeatureDecimation"]:
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
                    ("name", "U20"),
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
        feature_decimation_level_metadata: Optional[object] = None,
    ) -> FeatureDecimationGroup:
        """
        Add a decimation level group to the feature run.

        Parameters
        ----------
        decimation_level_name : str
            Name for the decimation level.
        feature_decimation_level_metadata : optional
            Metadata for the decimation level. Default is None.

        Returns
        -------
        FeatureDecimationGroup
            Newly created decimation level group.

        Examples
        --------
        >>> fc_run = FeatureFCRunGroup(h5_group)
        >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)
        >>> print(decimation.name)
        'level_0'
        """

        return self._add_group(
            decimation_level_name,
            FeatureDecimationGroup,
            group_metadata=feature_decimation_level_metadata,
            match="id",
        )

    def get_decimation_level(
        self, decimation_level_name: str
    ) -> FeatureDecimationGroup:
        """
        Retrieve a decimation level group by name.

        Parameters
        ----------
        decimation_level_name : str
            Name of the decimation level to retrieve.

        Returns
        -------
        FeatureDecimationGroup
            The requested decimation level group.

        Raises
        ------
        MTH5Error
            If the decimation level does not exist.

        Examples
        --------
        >>> fc_run = FeatureFCRunGroup(h5_group)
        >>> decimation = fc_run.get_decimation_level('level_0')
        """
        return self._get_group(decimation_level_name, FeatureDecimationGroup)

    def remove_decimation_level(self, decimation_level_name: str) -> None:
        """
        Remove a decimation level from the feature run.

        Parameters
        ----------
        decimation_level_name : str
            Name of the decimation level to remove.

        Raises
        ------
        MTH5Error
            If the decimation level does not exist.

        Examples
        --------
        >>> fc_run = FeatureFCRunGroup(h5_group)
        >>> fc_run.remove_decimation_level('level_0')
        """

        self._remove_group(decimation_level_name)

    def update_metadata(self) -> None:
        """
        Update metadata from all decimation levels.

        Scans all decimation levels and updates the run-level metadata with
        aggregated information including time ranges.

        Examples
        --------
        >>> fc_run = FeatureFCRunGroup(h5_group)
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

    # def supports_aurora_processing_config(
    #     self, processing_config, remote
    # ) -> bool:
    #     """

    #     An "all-or-nothing" check: Return True if every (valid) decimation needed to satisfy the processing_config
    #      is available in the FCGroup (self) otherwise return False (and we will build all FCs).

    #     Logic:
    #     1. Get a list of all fc groups in the FCGroup (self)
    #     2. Loop the processing_config decimations, checking if there is a corresponding, already built FCDecimation
    #      in the FCGroup.

    #     Parameters
    #     ----------
    #     processing_config: aurora.config.metadata.processing.Processing
    #     remote: bool

    #     Returns
    #     -------

    #     """
    #     pre_existing_fc_decimation_ids_to_check = self.groups_list
    #     levels_present = np.full(processing_config.num_decimation_levels, False)
    #     for i, dec_level in enumerate(processing_config.decimations):

    #         # Quit checking if dec_level wasn't there
    #         if i > 0:
    #             if not levels_present[i - 1]:
    #                 return False

    #         # iterate over existing decimations
    #         for fc_decimation_id in pre_existing_fc_decimation_ids_to_check:
    #             fc_dec_group = self.get_decimation_level(fc_decimation_id)
    #             fc_decimation = fc_dec_group.metadata
    #             levels_present[i] = fc_decimation.has_fcs_for_aurora_processing(
    #                 dec_level, remote
    #             )

    #             if levels_present[i]:
    #                 pre_existing_fc_decimation_ids_to_check.remove(
    #                     fc_decimation_id
    #                 )  # no need to check this one again
    #                 break  # break inner for-loop over decimations

    #     return levels_present.all()


class FeatureDecimationGroup(BaseGroup):
    """
    Container for a single decimation level with multiple Fourier Coefficient channels.

    This class manages Fourier Coefficient data organized by frequency, time, and channel.
    Data is assumed to be uniformly sampled in both frequency and time domains.

    Hierarchy
    ---------
    FeatureDecimationGroup -> FeatureChannelDataset (multiple channels)

    Data Assumptions
    ----------------
    1. Data are uniformly sampled in frequency domain
    2. Data are uniformly sampled in time domain
    3. FFT moving window has uniform step size

    Attributes
    ----------
    start time : datetime
        Start time of the decimation level
    end time : datetime
        End time of the decimation level
    channels : list
        List of channel names in this decimation level
    decimation_factor : int
        Factor by which data was decimated
    decimation_level : int
        Level index in decimation hierarchy
    decimation_sample_rate : float
        Sample rate after decimation (Hz)
    method : str
        Method used (FFT, wavelet, etc.)
    anti_alias_filter : optional
        Anti-aliasing filter used
    prewhitening_type : optional
        Type of prewhitening applied
    harmonics_kept : list or 'all'
        Harmonic indices kept in the data
    window : dict
        Window parameters (length, overlap, type, sample rate)
    bands : list
        Frequency bands in the data

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object for this FeatureDecimationGroup.
    decimation_level_metadata : optional
        Metadata for the decimation level. Default is None.
    **kwargs
        Additional keyword arguments passed to BaseGroup.

    Examples
    --------
    >>> decimation = FeatureDecimationGroup(h5_group, metadata)
    >>> channel = decimation.add_channel('Ex', fc_data=fc_array, fc_metadata=ch_metadata)
    """

    def __init__(
        self,
        group: h5py.Group,
        decimation_level_metadata: Optional[object] = None,
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

            - name : str
                Channel name
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
        >>> decimation = FeatureDecimationGroup(h5_group)
        >>> summary = decimation.channel_summary
        >>> print(summary[['name', 'n_frequency', 'n_windows']])
        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FCChannel"]:
                    ch_list.append(
                        (
                            group.attrs["name"],
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
                    ("name", "U20"),
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
        >>> decimation = FeatureDecimationGroup(h5_group)
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
        data_array: xr.DataArray | xr.Dataset,
        sample_rate_decimation_level: float,
    ) -> None:
        """
        Load Fourier Coefficient data from an xarray DataArray or Dataset.

        Automatically extracts metadata (time, frequency, units) from the xarray
        object and creates appropriate FeatureChannelDataset instances for each
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
        Automatically handles both (time, frequency) and (frequency, time) dimension ordering.
        Units are extracted from xarray attributes if available.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> decimation = FeatureDecimationGroup(h5_group)

        Create sample xarray data:

        >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
        >>> freqs = np.linspace(0.01, 100, 256)
        >>> data_array = np.random.randn(len(times), len(freqs)) + \\
        ...              1j * np.random.randn(len(times), len(freqs))
        >>> xr_data = xr.DataArray(
        ...     data_array,
        ...     dims=['time', 'frequency'],
        ...     coords={'time': times, 'frequency': freqs},
        ...     name='Ex',
        ...     attrs={'units': 'mV/km'}
        ... )

        Load into decimation group:

        >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)
        """

        if not isinstance(data_array, (xr.Dataset, xr.DataArray)):
            msg = f"Must input a xarray Dataset or DataArray not {type(data_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
        ch_metadata = FeatureDecimationChannel()
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
                ch_metadata.name = ch
                self._set_prewhitening_metadata(ch_metadata, data_array[ch])
                if ch in self.channel_summary.name.to_list():
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
        channel_metadata: FeatureDecimationChannel,
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

    def to_xarray(self, channels: Optional[list] = None) -> xr.Dataset:
        """
        Create an xarray Dataset from Fourier Coefficient channels.

        If no channels are specified, all channels in the decimation level
        are included. Each channel becomes a data variable in the resulting Dataset.

        Parameters
        ----------
        channels : list, optional
            List of channel names to include. If None, all channels are used.
            Default is None.

        Returns
        -------
        xr.Dataset
            xarray Dataset with channels as data variables and 'time' and
            'frequency' as shared coordinates.

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
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
        ch_name: str | list,
    ) -> None:
        """
        Load Fourier Coefficient data from a numpy array.

        Assumes array shape is either (n_frequencies, n_windows) for a single
        channel or (n_channels, n_frequencies, n_windows) for multiple channels.

        Parameters
        ----------
        nd_array : np.ndarray
            Input numpy array containing coefficient data.
        ch_name : str or list
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
        >>> decimation = FeatureDecimationGroup(h5_group)

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
        fc_data: Optional[np.ndarray | xr.DataArray | xr.Dataset | pd.DataFrame] = None,
        fc_metadata: Optional[FeatureDecimationChannel] = None,
        max_shape: tuple = (None, None),
        chunks: bool = True,
        dtype: type = complex,
        **kwargs,
    ) -> FeatureChannelDataset:
        """
        Add a Fourier Coefficient channel to the decimation level.

        Creates a new FeatureChannelDataset for a single channel at a single
        decimation level. Input data can be provided as numpy array, xarray,
        DataFrame, or created empty.

        Parameters
        ----------
        fc_name : str
            Name for the Fourier Coefficient channel.
        fc_data : np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, optional
            Input data. Can be numpy array (time, frequency) or xarray/DataFrame
            format. Default is None (creates empty dataset).
        fc_metadata : FeatureDecimationChannel, optional
            Metadata for the channel. Default is None.
        max_shape : tuple, default=(None, None)
            Maximum shape for HDF5 dataset dimensions (expandable if None).
        chunks : bool, default=True
            Whether to use HDF5 chunking.
        dtype : type, default=complex
            Data type for the dataset (e.g., complex, float, int).
        **kwargs
            Additional keyword arguments for HDF5 dataset creation.

        Returns
        -------
        FeatureChannelDataset
            Newly created FeatureChannelDataset object.

        Raises
        ------
        TypeError
            If fc_data type is not supported or metadata type mismatch.
        RuntimeError or OSError
            If channel already exists (will return existing channel).

        Notes
        -----
        Data layout assumes (time, frequency) organization:

        - time index: window start times
        - frequency index: harmonic indices or float values
        - data: complex Fourier coefficients

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
        >>> metadata = FeatureDecimationChannel(name='Ex')

        Create from numpy array:

        >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
        >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

        Create empty channel (expandable):

        >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)
        """

        fc_name = validate_name(fc_name)

        if fc_metadata is None:
            fc_metadata = FeatureDecimationChannel(name=fc_name)
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

            fc_dataset = FeatureChannelDataset(dataset, dataset_metadata=fc_metadata)
        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
            msg = (
                f"estimate {fc_metadata.name} already exists, returning existing group."
            )
            self.logger.debug(msg)

            fc_dataset = self.get_channel(fc_metadata.name)
        return fc_dataset

    def get_channel(self, fc_name: str) -> FeatureChannelDataset:
        """
        Retrieve a Fourier Coefficient channel by name.

        Parameters
        ----------
        fc_name : str
            Name of the channel to retrieve.

        Returns
        -------
        FeatureChannelDataset
            The requested FeatureChannelDataset object.

        Raises
        ------
        MTH5Error
            If the channel does not exist.

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
        >>> channel = decimation.get_channel('Ex')
        >>> data = channel.to_numpy()
        """
        fc_name = validate_name(fc_name)

        try:
            fc_dataset = self.hdf5_group[fc_name]
            fc_metadata = FeatureDecimationChannel(**dict(fc_dataset.attrs))
            return FeatureChannelDataset(fc_dataset, dataset_metadata=fc_metadata)
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

        Deletes the channel from the HDF5 file. Note that this removes the
        reference but does not reduce file size.

        Parameters
        ----------
        fc_name : str
            Name of the channel to remove.

        Raises
        ------
        MTH5Error
            If the channel does not exist.

        Notes
        -----
        To reduce HDF5 file size, copy desired data to a new file.

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
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
        Update metadata from all channels in the decimation level.

        Scans all channels and updates the decimation-level metadata with
        aggregated information including time ranges and sampling rates.

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
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

    def add_weights(
        self,
        weight_name: str,
        weight_data: Optional[np.ndarray] = None,
        weight_metadata: Optional[object] = None,
        max_shape: tuple = (None, None, None),
        chunks: bool = True,
        **kwargs,
    ) -> None:
        """
        Add weight or masking data for Fourier Coefficients.

        Creates a dataset to store weights or masks for quality control,
        frequency band selection, or time window filtering.

        Parameters
        ----------
        weight_name : str
            Name for the weight dataset.
        weight_data : np.ndarray, optional
            Weight values. Default is None.
        weight_metadata : optional
            Metadata for the weight dataset. Default is None.
        max_shape : tuple, default=(None, None, None)
            Maximum shape for expandable dimensions.
        chunks : bool, default=True
            Whether to use HDF5 chunking.
        **kwargs
            Additional keyword arguments for HDF5 dataset creation.

        Notes
        -----
        Weight datasets can track:

        - weight_channel: Per-channel weights
        - weight_band: Per-frequency-band weights
        - weight_time: Per-time-window weights

        This method is a placeholder for future implementation.

        Examples
        --------
        >>> decimation = FeatureDecimationGroup(h5_group)
        >>> decimation.add_weights('coherency_weights', weight_data=weights)
        """
