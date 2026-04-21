# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:02:16 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import weakref
from typing import Optional

import h5py
import numpy as np
import xarray as xr
from loguru import logger
from mt_metadata.features import FeatureDecimationChannel
from mt_timeseries.ts_helpers import make_dt_coordinates

from mth5.helpers import add_attributes_to_metadata_class_pydantic, to_numpy_type
from mth5.utils.exceptions import MTH5Error


# =============================================================================


class FeatureChannelDataset:
    """
    Container for multi-dimensional Fourier Coefficients organized by time and frequency.

    This class manages Fourier Coefficient data with frequency band organization,
    similar to FCDataset but with enhanced band tracking capabilities. The data array
    is organized with the following assumptions:

    1. Data are grouped into frequency bands
    2. Data are uniformly sampled in time (uniform FFT moving window step size)

    The dataset tracks temporal evolution of frequency content across multiple windows,
    making it suitable for time-frequency analysis of geophysical signals.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset containing the Fourier coefficient data.
    dataset_metadata : FeatureDecimationChannel, optional
        Metadata for the dataset. See :class:`mt_metadata.features.FeatureDecimationChannel`.
        If provided, must be of the same type as the internal metadata class.
        Default is None.
    **kwargs
        Additional keyword arguments for future extensibility.

    Attributes
    ----------
    hdf5_dataset : h5py.Dataset
        Reference to the HDF5 dataset.
    metadata : FeatureDecimationChannel
        Metadata container with the following attributes:

        - name : str
            Dataset name
        - time_period.start : datetime
            Start time of the data acquisition
        - time_period.end : datetime
            End time of the data acquisition
        - sample_rate_window_step : float
            Sample rate of the time window stepping (Hz)
        - frequency_min : float
            Minimum frequency in the band (Hz)
        - frequency_max : float
            Maximum frequency in the band (Hz)
        - units : str
            Physical units of the coefficient data
        - component : str
            Component identifier (e.g., 'Ex', 'Hy')
        - sample_rate_decimation_level : int
            Decimation level applied to acquire this data

    Raises
    ------
    MTH5Error
        If dataset_metadata type does not match the expected FeatureDecimationChannel type.

    Examples
    --------
    >>> import h5py
    >>> from mt_metadata.features import FeatureDecimationChannel
    >>> from mth5.groups.feature_dataset import FeatureChannelDataset

    Create a feature dataset from an HDF5 group:

    >>> with h5py.File('data.h5', 'r') as f:
    ...     h5_dataset = f['feature_group']['Ex']
    ...     feature = FeatureChannelDataset(h5_dataset)
    ...     print(f"Time windows: {feature.n_windows}")
    ...     print(f"Frequencies: {feature.n_frequencies}")

    Access time and frequency arrays:

    >>> time_array = feature.time
    >>> freq_array = feature.frequency
    >>> data_array = feature.to_numpy()
    """

    def __init__(
        self,
        dataset: h5py.Dataset,
        dataset_metadata: Optional[FeatureDecimationChannel] = None,
        **kwargs,
    ) -> None:
        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = add_attributes_to_metadata_class_pydantic(
            FeatureDecimationChannel
        )
        self.metadata.hdf5_reference = self.hdf5_dataset.ref
        self.metadata.mth5_type = self._class_name
        # if the input data set already has filled attributes, namely if the
        # channel data already exists then read them in with our writing back
        if "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.metadata.from_dict(
                {self.hdf5_dataset.attrs["mth5_type"]: dict(self.hdf5_dataset.attrs)}
            )
        # if metadata is input, make sure that its the same class type amd write
        # to the hdf5 dataset
        if dataset_metadata is not None:
            if not isinstance(self.metadata, type(dataset_metadata)):
                msg = (
                    f"metadata must be type metadata.{self._class_name} not "
                    "{type(dataset_metadata)}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(dataset_metadata.to_dict())
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            try:
                self.write_metadata()
            except (RuntimeError, KeyError, OSError):
                # file is read only
                pass

        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

    def __str__(self) -> str:
        """
        String representation of the FeatureChannelDataset.

        Returns
        -------
        str
            JSON representation of the metadata.
        """
        return self.metadata.to_json()

    def __repr__(self) -> str:
        """
        Official string representation of the FeatureChannelDataset.

        Returns
        -------
        str
            JSON representation of the metadata.
        """
        return self.__str__()

    @property
    def _class_name(self) -> str:
        """
        Extract the class name prefix by removing 'Dataset' suffix.

        Returns
        -------
        str
            Class name without the 'Dataset' suffix.
        """
        return self.__class__.__name__.split("Dataset")[0]

    def read_metadata(self) -> None:
        """
        Read metadata from the HDF5 file into the metadata container.

        This method loads all attributes from the HDF5 dataset into the
        metadata container, enabling validation and type checking.

        Examples
        --------
        >>> feature.read_metadata()
        >>> print(feature.metadata.component)
        'Ex'
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
        Write metadata from the metadata container to the HDF5 attributes.

        This method serializes the metadata container and writes all metadata
        as attributes to the HDF5 dataset. Raises exceptions are caught for
        read-only files.

        Examples
        --------
        >>> feature.metadata.component = 'Ey'
        >>> feature.write_metadata()
        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.hdf5_dataset.attrs.create(key, value)

    @property
    def n_windows(self) -> int:
        """
        Get the number of time windows in the dataset.

        Returns
        -------
        int
            Number of time windows (first dimension of the dataset).
        """
        return self.hdf5_dataset.shape[0]

    @property
    def time(self) -> np.ndarray:
        """
        Get the time array for each window.

        Returns an array of datetime64 values representing the start time
        of each time window. The time spacing is determined by the sample
        rate of the window stepping.

        Returns
        -------
        np.ndarray
            Array of datetime64 values with shape (n_windows,) representing
            the start time of each window.

        Examples
        --------
        >>> time_array = feature.time
        >>> print(time_array.shape)
        (100,)
        >>> print(time_array[0])
        numpy.datetime64('2023-01-01T00:00:00')
        """

        return make_dt_coordinates(
            self.metadata.time_period.start,
            1.0 / self.metadata.sample_rate_window_step,
            self.n_windows,
            end_time=self.metadata.time_period.end,
        )

    @property
    def n_frequencies(self) -> int:
        """
        Get the number of frequency bins in the dataset.

        Returns
        -------
        int
            Number of frequency bins (second dimension of the dataset).
        """
        return self.hdf5_dataset.shape[1]

    @property
    def frequency(self) -> np.ndarray:
        """
        Get the frequency array for the dataset.

        Returns a linearly-spaced frequency array from frequency_min to
        frequency_max with n_frequencies points.

        Returns
        -------
        np.ndarray
            Array of float64 frequencies in Hz with shape (n_frequencies,).

        Examples
        --------
        >>> freq_array = feature.frequency
        >>> print(freq_array.shape)
        (256,)
        >>> print(f"Frequency range: {freq_array[0]:.2f} - {freq_array[-1]:.2f} Hz")
        Frequency range: 0.01 - 100.00 Hz
        """
        return np.linspace(
            self.metadata.frequency_min,
            self.metadata.frequency_max,
            self.n_frequencies,
        )

    def replace_dataset(self, new_data_array: np.ndarray) -> None:
        """
        Replace the entire HDF5 dataset with new data.

        This method resizes the HDF5 dataset as needed and replaces all data.
        The input array must have the same dtype as the existing dataset.

        Parameters
        ----------
        new_data_array : np.ndarray
            New data array to replace the existing dataset. Will be converted
            to numpy array if necessary.

        Raises
        ------
        TypeError
            If input cannot be converted to a numpy array or has incompatible shape.

        Examples
        --------
        >>> import numpy as np
        >>> new_data = np.random.randn(100, 256)
        >>> feature.replace_dataset(new_data)
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

    def to_xarray(self) -> xr.DataArray:
        """
        Convert the feature dataset to an xarray DataArray.

        Returns an xarray DataArray with proper time and frequency coordinates,
        metadata attributes, and component naming. The entire dataset is loaded
        into memory.

        Returns
        -------
        xr.DataArray
            DataArray with dimensions ['time', 'frequency'] and coordinates
            matching the dataset's time and frequency arrays.

        Notes
        -----
        Metadata stored in xarray attributes will not be validated if modified.
        The full dataset is loaded into memory; use with caution for large datasets.

        Examples
        --------
        >>> xr_data = feature.to_xarray()
        >>> print(xr_data.dims)
        ('time', 'frequency')
        >>> print(xr_data.name)
        'Ex'
        >>> subset = xr_data.sel(time=slice('2023-01-01', '2023-01-02'))
        """

        return xr.DataArray(
            data=self.hdf5_dataset[()],
            dims=["time", "frequency"],
            name=self.metadata.component,
            coords=[
                ("time", self.time),
                ("frequency", self.frequency),
            ],
            attrs=self.metadata.to_dict(single=True),
        )

    def to_numpy(self) -> np.ndarray:
        """
        Convert the feature dataset to a numpy array.

        Returns the dataset as a numpy array by loading it from the HDF5 file
        into memory. The array shape is (n_windows, n_frequencies).

        Returns
        -------
        np.ndarray
            Numpy array containing all feature data with shape
            (n_windows, n_frequencies).

        Examples
        --------
        >>> data = feature.to_numpy()
        >>> print(data.shape)
        (100, 256)
        >>> print(data.dtype)
        complex128
        >>> mean_amplitude = np.abs(data).mean()
        """

        return self.hdf5_dataset[()]

    def from_numpy(self, new_estimate: np.ndarray) -> None:
        """
        Load data from a numpy array into the HDF5 dataset.

        This method updates the HDF5 dataset with new data from a numpy array.
        The input array must match the dataset's dtype. The HDF5 dataset will
        be resized if necessary to accommodate the new data.

        Parameters
        ----------
        new_estimate : np.ndarray
            Numpy array to write to the HDF5 dataset. Must have compatible
            dtype with the existing dataset.

        Raises
        ------
        TypeError
            If input array dtype does not match the HDF5 dataset dtype or
            if input cannot be converted to numpy array.

        Notes
        -----
        The variable 'data' is a builtin in numpy and cannot be used as a parameter name.

        Examples
        --------
        >>> import numpy as np
        >>> new_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
        >>> feature.from_numpy(new_data)
        >>> loaded_data = feature.to_numpy()
        >>> assert loaded_data.shape == new_data.shape
        """

        if not isinstance(new_estimate, np.ndarray):
            try:
                new_estimate = np.array(new_estimate)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_estimate)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if new_estimate.dtype != self.hdf5_dataset.dtype:
            msg = (
                f"Input array must be type {new_estimate.dtype} not "
                "{self.hdf5_dataset.dtype}"
            )
            self.logger.error(msg)
            raise TypeError(msg)
        if new_estimate.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_estimate.shape)
        self.hdf5_dataset[...] = new_estimate

    def from_xarray(
        self,
        data: xr.DataArray,
        sample_rate_decimation_level: int,
    ) -> None:
        """
        Load data and metadata from an xarray DataArray.

        This method updates both the HDF5 dataset and metadata from an xarray
        DataArray. It extracts time coordinates, frequency range, and component
        information from the DataArray and its attributes.

        Parameters
        ----------
        data : xr.DataArray
            Input xarray DataArray with 'time' and 'frequency' coordinates.
            Expected dimensions are ['time', 'frequency'].
        sample_rate_decimation_level : int
            Decimation level applied to the original data to produce this
            feature dataset (integer >= 1).

        Notes
        -----
        Metadata stored in xarray attributes will be extracted and written to
        the HDF5 file. The full dataset is loaded into memory during this process.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np

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

        Load into feature dataset:

        >>> feature.from_xarray(xr_data, sample_rate_decimation_level=2)
        >>> print(feature.metadata.component)
        'Ex'
        """
        self.metadata.time_period.start = data.time[0].values
        self.metadata.time_period.end = data.time[-1].values

        self.metadata.sample_rate_decimation_level = sample_rate_decimation_level
        self.metadata.frequency_min = data.coords["frequency"].data.min()
        self.metadata.frequency_max = data.coords["frequency"].data.max()
        step_size = data.coords["time"].data[1] - data.coords["time"].data[0]
        self.metadata.sample_rate_window_step = step_size / np.timedelta64(1, "s")
        self.metadata.component = data.name
        try:
            self.metadata.units = data.units
        except AttributeError:
            self.logger.debug("Could not find 'units' in xarray")
        self.write_metadata()

        self.from_numpy(data.to_numpy())
