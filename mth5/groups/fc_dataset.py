# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:02:16 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import weakref
from typing import Any

import h5py
import numpy as np
import xarray as xr
from loguru import logger
from mt_metadata.processing.fourier_coefficients import FCChannel
from mt_timeseries.ts_helpers import make_dt_coordinates

from mth5.helpers import add_attributes_to_metadata_class_pydantic, to_numpy_type
from mth5.utils.exceptions import MTH5Error


# =============================================================================


class FCChannelDataset:
    """
    Container for Fourier coefficients (FC) from windowed FFT analysis.

    Holds multi-dimensional Fourier coefficient data representing time-frequency
    analysis results. Data is uniformly sampled in both frequency (via harmonic
    index) and time (via uniform FFT window step size).

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset containing the Fourier coefficient data.
    dataset_metadata : FCChannel | None, optional
        Metadata object containing FC channel properties like start time,
        end time, sample rates, units, and frequency method. If provided,
        metadata will be written to HDF5 attributes. Defaults to None.
    **kwargs : Any
        Additional keyword arguments (reserved for future use).

    Attributes
    ----------
    hdf5_dataset : h5py.Dataset
        Weak reference to the HDF5 dataset.
    metadata : FCChannel
        Metadata container for the Fourier coefficients.
    logger : loguru.logger
        Logger instance for reporting messages.

    Raises
    ------
    MTH5Error
        If dataset_metadata is provided but is not of type FCChannel.
    TypeError
        If input data cannot be converted to numpy array or has
        incompatible dtype/shape.

    Notes
    -----
    The data array has shape (n_windows, n_frequencies) where:
    - n_windows: Number of time windows in the FFT moving window analysis
    - n_frequencies: Number of frequency bins determined by window size

    Data is typically complex-valued representing Fourier coefficients.
    Time windows are uniformly spaced with interval 1/sample_rate_window_step.
    Frequencies are uniformly spaced from frequency_min to frequency_max.

    Metadata includes:
    - Time period (start and end)
    - Acquisition and decimated sample rates
    - Window sample rate (delta_t within window)
    - Units
    - Frequency method (integer harmonic index calculation)
    - Component name (channel designation)

    Examples
    --------
    Create an FC dataset from HDF5 group:

    >>> import h5py
    >>> import numpy as np
    >>> from mt_metadata.processing.fourier_coefficients import FCChannel
    >>> with h5py.File('fc.h5', 'w') as f:
    ...     # Create 2D array: 50 time windows, 256 frequencies
    ...     data = np.random.rand(50, 256) + 1j * np.random.rand(50, 256)
    ...     dset = f.create_dataset('Ex', data=data, dtype=np.complex128)
    ...     # Create FCChannelDataset
    ...     fc = FCChannelDataset(dset, write_metadata=True)

    Convert to xarray and access time-frequency data:

    >>> xr_data = fc.to_xarray()
    >>> print(xr_data.dims)  # ('time', 'frequency')
    >>> # Access data at specific time and frequency
    >>> subset = xr_data.sel(time='2023-01-01T12:00:00', method='nearest')

    Inspect properties:

    >>> print(f"Windows: {fc.n_windows}, Frequencies: {fc.n_frequencies}")
    >>> print(f"Frequency range: {fc.frequency.min():.2f}-{fc.frequency.max():.2f} Hz")

    """

    def __init__(
        self,
        dataset: h5py.Dataset,
        dataset_metadata: FCChannel | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an FCChannelDataset.

        Parameters
        ----------
        dataset : h5py.Dataset
            HDF5 dataset for storing Fourier coefficient data.
        dataset_metadata : FCChannel | None, optional
            Metadata object. If provided, updates internal metadata and
            writes to HDF5 (unless file is read-only). Defaults to None.
        **kwargs : Any
            Additional keyword arguments (reserved for future use).

        Raises
        ------
        MTH5Error
            If dataset_metadata type doesn't match FCChannel.

        Notes
        -----
        Metadata is automatically read from HDF5 attributes if 'mth5_type'
        attribute exists. Write operations are wrapped in try-except to
        gracefully handle read-only files.

        Examples
        --------
        Create and initialize an FC dataset:

        >>> import h5py
        >>> import numpy as np
        >>> with h5py.File('fc.h5', 'w') as f:
        ...     data = np.random.rand(20, 128) + 1j * np.random.rand(20, 128)
        ...     dset = f.create_dataset('Ex', data=data)
        ...     fc = FCChannelDataset(dset)  # Auto-initialize metadata

        """
        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        metadata_obj = FCChannel
        self.metadata = add_attributes_to_metadata_class_pydantic(metadata_obj)
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
        Return string representation of the FC dataset as JSON.

        Returns
        -------
        str
            JSON representation of the FC metadata.

        Examples
        --------
        >>> fc_str = str(fc)
        >>> print(fc_str[:50])  # Print first 50 characters
        {"fcchannel": {"component": "Ex", ...

        """
        return self.metadata.to_json()

    def __repr__(self) -> str:
        """
        Return official string representation of the FC dataset.

        Returns
        -------
        str
            JSON representation of the FC metadata.

        Examples
        --------
        >>> repr(fc) == str(fc)
        True

        """
        return self.__str__()

    @property
    def _class_name(self) -> str:
        """
        Extract the class name without 'Dataset' suffix.

        Returns
        -------
        str
            Class name with 'Dataset' suffix removed.

        Examples
        --------
        >>> fc._class_name
        'FCChannel'

        """
        return self.__class__.__name__.split("Dataset")[0]

    def read_metadata(self) -> None:
        """
        Read metadata from HDF5 attributes into metadata container.

        Reads all attributes from the HDF5 dataset and loads them into
        the internal metadata object for validation and access.

        Returns
        -------
        None

        Notes
        -----
        This is automatically called during initialization if 'mth5_type'
        attribute exists in the HDF5 dataset.

        Examples
        --------
        Reload metadata from HDF5 after external modification:

        >>> # Metadata was modified in HDF5
        >>> fc.read_metadata()  # Reload changes
        >>> print(fc.metadata.component)  # Access updated component

        """
        self.metadata.from_dict({self._class_name: dict(self.hdf5_dataset.attrs)})

    def write_metadata(self) -> None:
        """
        Write metadata from container to HDF5 dataset attributes.

        Converts the pydantic metadata model to a dictionary and writes
        each field as an HDF5 attribute. Values are converted to appropriate
        numpy types for compatibility. Always ensures 'mth5_type' attribute
        is set to 'FCChannel'.

        Returns
        -------
        None

        Notes
        -----
        All existing attributes with the same names will be overwritten.
        This is called automatically during initialization and after
        metadata updates. Read-only files will silently skip writes.

        Examples
        --------
        Save updated metadata to HDF5:

        >>> fc.metadata.component = "Ey"
        >>> fc.write_metadata()  # Persist to file
        >>> # Verify write
        >>> print(fc.hdf5_dataset.attrs['component'])
        b'Ey'

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.hdf5_dataset.attrs.create(key, value)

        # Add the mth5_type attribute that is expected by channel_summary
        if "mth5_type" not in self.hdf5_dataset.attrs:
            self.hdf5_dataset.attrs.create("mth5_type", "FCChannel")

    @property
    def n_windows(self) -> int:
        """
        Number of time windows in the FFT analysis.

        Returns
        -------
        int
            Number of time windows (first dimension of data array).

        Notes
        -----
        This corresponds to the number of rows in the 2D spectrogram data.
        Each window represents a uniform time interval determined by the
        window step size (1/sample_rate_window_step).

        Examples
        --------
        >>> print(f"Time windows: {fc.n_windows}")
        Time windows: 50

        """
        return self.hdf5_dataset.shape[0]

    @property
    def time(self) -> np.ndarray:
        """
        Time array including the start of each time window.

        Generates uniformly spaced time coordinates based on the start time,
        window step rate, and number of windows. Uses metadata time period
        to determine bounds.

        Returns
        -------
        np.ndarray
            Array of datetime64 values for each window start time.

        Notes
        -----
        Time coordinates are generated using make_dt_coordinates, which
        ensures consistency between specified start/end times and the
        number of windows.

        Examples
        --------
        Access time array for time-based indexing:

        >>> time_array = fc.time
        >>> print(time_array.shape)  # (n_windows,)
        >>> print(time_array[0])  # First window time
        2023-01-01T00:00:00.000000

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
        Number of frequency bins in the Fourier analysis.

        Returns
        -------
        int
            Number of frequency bins (second dimension of data array).

        Notes
        -----
        This corresponds to the number of columns in the 2D spectrogram data.
        Determined by the FFT window size and relates to the frequency
        resolution of the analysis.

        Examples
        --------
        >>> print(f"Frequency bins: {fc.n_frequencies}")
        Frequency bins: 256

        """
        return self.hdf5_dataset.shape[1]

    @property
    def frequency(self) -> np.ndarray:
        """
        Frequency array from metadata frequency bounds.

        Generates uniformly spaced frequency coordinates based on the
        metadata frequency range and number of frequency bins.

        Returns
        -------
        np.ndarray
            Array of frequency values, linearly spaced from frequency_min
            to frequency_max.

        Notes
        -----
        Frequencies represent harmonic indices or actual frequency values
        depending on the frequency method specified in metadata.
        Spacing is determined by n_frequencies bins over the range.

        Examples
        --------
        Access frequency array for frequency-based indexing:

        >>> freq_array = fc.frequency
        >>> print(freq_array.shape)  # (n_frequencies,)
        >>> print(f"Frequency range: {freq_array.min():.2f} to {freq_array.max():.2f} Hz")
        Frequency range: 0.00 to 64.00 Hz

        """
        return np.linspace(
            self.metadata.frequency_min,
            self.metadata.frequency_max,
            self.n_frequencies,
        )

    def replace_dataset(self, new_data_array: np.ndarray) -> None:
        """
        Replace entire dataset with new data.

        Resizes the HDF5 dataset if necessary and replaces all data.
        Converts input to numpy array if needed.

        Parameters
        ----------
        new_data_array : np.ndarray
            New FC data to store. Should have shape (n_windows, n_frequencies)
            and typically complex-valued.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If input cannot be converted to numpy array.

        Notes
        -----
        If new data has different shape, HDF5 dataset will be resized.
        This is generally safe but may fragment the HDF5 file.

        Examples
        --------
        Replace FC data with new analysis results:

        >>> import numpy as np
        >>> new_fc = np.random.rand(30, 256) + 1j * np.random.rand(30, 256)
        >>> fc.replace_dataset(new_fc)
        >>> print(fc.to_numpy().shape)
        (30, 256)

        Replace with data from list (auto-converted to array):

        >>> data_list = [[[1+1j, 2+2j]], [[3+3j, 4+4j]]] * 15
        >>> fc.replace_dataset(data_list)
        >>> fc.to_numpy().shape
        (30, 2)

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
        Convert FC data to xarray DataArray.

        Creates an xarray DataArray with proper coordinates for time and
        frequency. Includes metadata as attributes.

        Returns
        -------
        xr.DataArray
            DataArray with dimensions (time, frequency) and coordinates
            from metadata and computed properties.

        Notes
        -----
        Metadata changes in xarray are not validated and will not be
        synchronized back to HDF5 without explicit call to from_xarray().
        Data is loaded entirely into memory.

        Examples
        --------
        Convert to xarray with automatic coordinates:

        >>> xr_data = fc.to_xarray()
        >>> print(xr_data.dims)
        ('time', 'frequency')
        >>> print(xr_data.shape)
        (50, 256)

        Select data by time and frequency range:

        >>> subset = xr_data.sel(
        ...     time=slice('2023-01-01T00:00:00', '2023-01-01T12:00:00'),
        ...     frequency=slice(0, 10)
        ... )
        >>> print(subset.shape)  # Subset shape

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
        Convert FC data to numpy array.

        Returns the HDF5 dataset as a numpy array. Data is loaded
        entirely into memory.

        Returns
        -------
        np.ndarray
            2D complex array with shape (n_windows, n_frequencies).

        Notes
        -----
        For large spectrograms, this loads all data into RAM. Consider using
        HDF5 slicing for memory-efficient access to subsets.

        Examples
        --------
        Get full FC data as numpy array:

        >>> data = fc.to_numpy()
        >>> print(data.shape)
        (50, 256)
        >>> print(data.dtype)
        complex128

        Access specific time window and frequency:

        >>> data = fc.to_numpy()
        >>> # Get first 10 windows, frequency bin 100
        >>> subset = data[:10, 100]
        >>> print(subset.shape)
        (10,)

        """
        return self.hdf5_dataset[()]

    def from_numpy(self, new_estimate: np.ndarray) -> None:
        """
        Load FC data from numpy array.

        Validates dtype and shape compatibility, resizes dataset if needed,
        and stores the data.

        Parameters
        ----------
        new_estimate : np.ndarray
            FC data to load. Should have shape (n_windows, n_frequencies).
            Typically complex-valued array.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If dtype doesn't match existing dataset or input cannot
            be converted to numpy array.

        Notes
        -----
        'data' is a built-in Python function and cannot be used as parameter name.
        The dataset will be resized if shape doesn't match.
        Dtype compatibility is strictly enforced.

        Examples
        --------
        Load FC data from numpy array:

        >>> import numpy as np
        >>> new_data = np.random.rand(25, 128) + 1j * np.random.rand(25, 128)
        >>> fc.from_numpy(new_data)
        >>> print(fc.to_numpy().shape)
        (25, 128)

        Load with magnitude and phase separation:

        >>> magnitude = np.random.rand(20, 256)
        >>> phase = np.random.rand(20, 256) * 2 * np.pi
        >>> fc_data = magnitude * np.exp(1j * phase)
        >>> fc.from_numpy(fc_data)

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
        sample_rate_decimation_level: int | float,
    ) -> None:
        """
        Load FC data from xarray DataArray.

        Updates metadata from xarray coordinates and attributes, then
        stores the data. Computes frequency and time parameters from
        the provided xarray object.

        Parameters
        ----------
        data : xr.DataArray
            DataArray containing FC data. Expected dimensions:
            (time, frequency).
        sample_rate_decimation_level : int | float
            Decimation level applied to original sample rate.
            Used to track processing history.

        Returns
        -------
        None

        Notes
        -----
        This will update time_period (start/end), frequency bounds,
        window step rate, decimation level, component name, and units
        from the xarray object. All changes are persisted to HDF5.

        Examples
        --------
        Load FC data from modified xarray:

        >>> xr_data = fc.to_xarray()
        >>> # Modify data (e.g., apply filter)
        >>> modified = xr_data * np.hamming(256)  # Apply frequency window
        >>> fc.from_xarray(modified, sample_rate_decimation_level=4)
        >>> print(fc.metadata.sample_rate_decimation_level)
        4

        Load with updated metadata from another analysis:

        >>> import xarray as xr
        >>> import pandas as pd
        >>> time_coords = pd.date_range('2023-01-01', periods=30, freq='1H')
        >>> freq_coords = np.arange(0, 128)
        >>> new_fc = xr.DataArray(
        ...     data=np.random.rand(30, 128) + 1j * np.random.rand(30, 128),
        ...     coords={'time': time_coords, 'frequency': freq_coords},
        ...     dims=['time', 'frequency'],
        ...     name='Ey',
        ...     attrs={'units': 'mV/km'}
        ... )
        >>> fc.from_xarray(new_fc, sample_rate_decimation_level=1)
        >>> print(fc.metadata.component)
        Ey

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
