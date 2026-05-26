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
from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
from mt_metadata.utils.validators import validate_attribute

from mth5.helpers import (
    add_attributes_to_metadata_class_pydantic,
    read_attrs_to_dict,
    to_numpy_type,
)
from mth5.utils.exceptions import MTH5Error


# =============================================================================


class EstimateDataset:
    """
    Container for statistical estimates of transfer functions.

    This class holds multi-dimensional statistical estimates for transfer
    functions with full metadata management. Estimates are stored as HDF5
    datasets with dimensions for period, output channels, and input channels.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset containing the statistical estimate data.
    dataset_metadata : mt_metadata.transfer_functions.tf.StatisticalEstimate, optional
        Metadata object for the estimate. If provided and write_metadata is True,
        the metadata will be written to the HDF5 attributes. Defaults to None.
    write_metadata : bool, optional
        If True, write metadata to the HDF5 dataset attributes. Defaults to True.
    **kwargs : Any
        Additional keyword arguments (reserved for future use).

    Attributes
    ----------
    hdf5_dataset : h5py.Dataset
        Weak reference to the HDF5 dataset.
    metadata : StatisticalEstimate
        Metadata container for the estimate.
    logger : loguru.logger
        Logger instance for reporting messages.

    Raises
    ------
    MTH5Error
        If dataset_metadata is provided but is not of type StatisticalEstimate
        or a compatible metadata class.
    TypeError
        If input data cannot be converted to numpy array or has wrong dtype/shape.

    Notes
    -----
    The estimate data is stored in 3D form with shape:
    (n_periods, n_output_channels, n_input_channels)

    Metadata is automatically synchronized between the pydantic model and
    HDF5 attributes on initialization and after any modifications.

    Examples
    --------
    Create an estimate dataset from an HDF5 group:

    >>> import h5py
    >>> import numpy as np
    >>> from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
    >>> # Create HDF5 file with estimate dataset
    >>> with h5py.File('estimate.h5', 'w') as f:
    ...     # Create dataset with shape (10 periods, 2 outputs, 2 inputs)
    ...     data = np.random.rand(10, 2, 2)
    ...     dset = f.create_dataset('estimate', data=data)
    ...     # Create EstimateDataset
    ...     est = EstimateDataset(dset, write_metadata=True)

    Convert estimate to xarray and back:

    >>> periods = np.logspace(-3, 3, 10)  # 10 periods from 1e-3 to 1e3 s
    >>> xr_data = est.to_xarray(periods)
    >>> # Modify xarray coordinates
    >>> new_xr = xr_data.rename({'output': 'new_output', 'input': 'new_input'})
    >>> est.from_xarray(new_xr)  # Load modified data back

    Access estimate data in different formats:

    >>> # Get numpy array
    >>> np_data = est.to_numpy()
    >>> print(np_data.shape)  # (10, 2, 2)
    >>> # Get xarray with proper coordinates
    >>> xr_data = est.to_xarray(periods)
    >>> print(xr_data.dims)  # ('period', 'output', 'input')

    """

    def __init__(
        self,
        dataset: h5py.Dataset,
        dataset_metadata: StatisticalEstimate | None = None,
        write_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an EstimateDataset.

        Parameters
        ----------
        dataset : h5py.Dataset
            HDF5 dataset for storing estimate data.
        dataset_metadata : StatisticalEstimate | None, optional
            Metadata object. If provided, updates internal metadata.
            Defaults to None.
        write_metadata : bool, optional
            Write metadata to HDF5 attributes. Defaults to True.
        **kwargs : Any
            Additional keyword arguments (reserved for future use).

        Raises
        ------
        MTH5Error
            If dataset_metadata type doesn't match expected metadata class.

        Examples
        --------
        Create and initialize an estimate dataset:

        >>> import h5py
        >>> import numpy as np
        >>> from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
        >>> with h5py.File('estimate.h5', 'w') as f:
        ...     data = np.random.rand(5, 2, 2)  # 5 periods, 2 outputs, 2 inputs
        ...     dset = f.create_dataset('estimate', data=data)
        ...     est = EstimateDataset(dset)  # Auto-initialize metadata

        """
        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        file_mode = getattr(getattr(self.hdf5_dataset, "file", None), "mode", "")
        can_write = "w" in file_mode or "+" in file_mode

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = add_attributes_to_metadata_class_pydantic(StatisticalEstimate)
        self.metadata.hdf5_reference = self.hdf5_dataset.ref
        self.metadata.mth5_type = validate_attribute(self._class_name)

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
            self.metadata.update(dataset_metadata)
            # self.metadata.hdf5_reference = self.hdf5_dataset.ref
            # self.metadata.mth5_type = self._class_name

            # Only write when metadata actually changed.
            if write_metadata and can_write and self._metadata_needs_write():
                self.write_metadata()
        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()) and can_write:
            self.write_metadata()

    def __str__(self) -> str:
        """
        Return string representation of the estimate as JSON.

        Returns
        -------
        str
            JSON representation of the estimate metadata.

        Examples
        --------
        >>> est_str = str(est)
        >>> print(est_str[:50])  # Print first 50 characters
        {"estimate": {"name": "estimate", ...

        """
        return self.metadata.to_json()

    def __repr__(self) -> str:
        """
        Return official string representation of the estimate.

        Returns
        -------
        str
            JSON representation of the estimate metadata.

        Examples
        --------
        >>> repr(est) == str(est)
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
        >>> est._class_name
        'Estimate'

        """
        return self.__class__.__name__.split("Dataset")[0]

    @staticmethod
    def _values_equal(existing_value: Any, expected_value: Any) -> bool:
        """Compare HDF5 attribute values, including ndarray attributes."""
        if isinstance(existing_value, np.ndarray) or isinstance(
            expected_value, np.ndarray
        ):
            return np.array_equal(
                np.asarray(existing_value), np.asarray(expected_value)
            )

        if isinstance(existing_value, np.generic):
            existing_value = existing_value.item()
        if isinstance(expected_value, np.generic):
            expected_value = expected_value.item()

        return existing_value == expected_value

    def _metadata_needs_write(self) -> bool:
        """Return ``True`` when in-memory metadata differs from dataset attrs."""
        expected_attrs = self.metadata.to_dict()[self.metadata._class_name.lower()]

        for key, value in expected_attrs.items():
            expected_value = to_numpy_type(value)
            if key not in self.hdf5_dataset.attrs:
                return True
            existing_value = self.hdf5_dataset.attrs[key]
            if not self._values_equal(existing_value, expected_value):
                return True

        return False

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
        >>> est.read_metadata()  # Reload changes
        >>> print(est.metadata.name)  # Access updated name

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
        Write metadata from container to HDF5 dataset attributes.

        Converts the pydantic metadata model to a dictionary and writes
        each field as an HDF5 attribute. Values are converted to appropriate
        numpy types for compatibility.

        Returns
        -------
        None

        Notes
        -----
        All existing attributes with the same names will be overwritten.
        This is called automatically during initialization and after
        metadata updates.

        Examples
        --------
        Save updated metadata to HDF5:

        >>> est.metadata.name = "Updated Estimate"
        >>> est.write_metadata()  # Persist to file
        >>> # Verify write
        >>> print(est.hdf5_dataset.attrs['name'])
        b'Updated Estimate'

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.hdf5_dataset.attrs.create(key, value)

    def replace_dataset(self, new_data_array: np.ndarray) -> None:
        """
        Replace entire dataset with new data.

        Resizes the HDF5 dataset if necessary and replaces all data.
        Converts input to numpy array if needed.

        Parameters
        ----------
        new_data_array : np.ndarray
            New estimate data to store. Should have shape
            (n_periods, n_output_channels, n_input_channels).

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
        Replace estimate with new data:

        >>> import numpy as np
        >>> new_estimate = np.random.rand(10, 2, 2)  # 10 periods, 2 channels
        >>> est.replace_dataset(new_estimate)
        >>> print(est.to_numpy().shape)
        (10, 2, 2)

        Replace with data from list (auto-converted to array):

        >>> data_list = [[[1, 2], [3, 4]]] * 5  # 5 periods
        >>> est.replace_dataset(data_list)
        >>> est.to_numpy().shape
        (5, 2, 2)

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

    def to_xarray(self, period: np.ndarray | list) -> xr.DataArray:
        """
        Convert estimate to xarray DataArray.

        Creates an xarray DataArray with proper coordinates for periods,
        output channels, and input channels. Includes metadata as attributes.

        Parameters
        ----------
        period : np.ndarray | list
            Period values for coordinate. Should have length equal to
            estimate first dimension (n_periods).

        Returns
        -------
        xr.DataArray
            DataArray with dimensions (period, output, input) and
            coordinates from metadata.

        Notes
        -----
        Metadata changes in xarray are not validated and will not be
        synchronized back to HDF5 without explicit call to from_xarray().
        Data is loaded entirely into memory.

        Examples
        --------
        Convert to xarray with logarithmic period spacing:

        >>> import numpy as np
        >>> periods = np.logspace(-2, 3, 10)  # 10 periods from 0.01 to 1000
        >>> xr_data = est.to_xarray(periods)
        >>> print(xr_data.dims)
        ('period', 'output', 'input')
        >>> print(xr_data.coords['period'].values)
        [1.00e-02 3.16e-02 ... 1.00e+03]

        Select data by period range:

        >>> subset = xr_data.sel(period=slice(0.1, 100))
        >>> print(subset.shape)
        (8, 2, 2)

        """
        return xr.DataArray(
            data=self.hdf5_dataset[()],
            dims=["period", "output", "input"],
            name=self.metadata.name,
            coords=[
                ("period", period),
                ("output", self.metadata.output_channels),
                ("input", self.metadata.input_channels),
            ],
            attrs=self.metadata.to_dict(single=True),
        )

    def to_numpy(self) -> np.ndarray:
        """
        Convert estimate to numpy array.

        Returns the HDF5 dataset as a numpy array. Data is loaded
        entirely into memory.

        Returns
        -------
        np.ndarray
            3D array with shape (n_periods, n_output_channels, n_input_channels).

        Notes
        -----
        For large estimates, this loads all data into RAM. Consider using
        HDF5 slicing for memory-efficient access.

        Examples
        --------
        Get full estimate as numpy array:

        >>> data = est.to_numpy()
        >>> print(data.shape)
        (10, 2, 2)
        >>> print(data.dtype)
        float64

        Access specific period and channels:

        >>> data = est.to_numpy()
        >>> # Get first 5 periods, output channel 0, input channel 1
        >>> subset = data[:5, 0, 1]
        >>> print(subset.shape)
        (5,)

        """
        return self.hdf5_dataset[()]

    def from_numpy(self, new_estimate: np.ndarray) -> None:
        """
        Load estimate data from numpy array.

        Validates dtype and shape compatibility, resizes dataset if needed,
        and stores the data.

        Parameters
        ----------
        new_estimate : np.ndarray
            Estimate data to load. Must be convertible to numpy array.
            Preferred shape: (n_periods, n_output_channels, n_input_channels).

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

        Examples
        --------
        Load estimate from numpy array:

        >>> import numpy as np
        >>> new_data = np.random.rand(5, 2, 2)
        >>> est.from_numpy(new_data)
        >>> print(est.to_numpy().shape)
        (5, 2, 2)

        Load with automatic dtype conversion:

        >>> float_data = np.array([[[1.0, 2.0]]], dtype=np.float64)
        >>> est.from_numpy(float_data)

        """
        if not isinstance(new_estimate, np.ndarray):
            try:
                new_estimate = np.array(new_estimate)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_estimate)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if new_estimate.dtype != self.hdf5_dataset.dtype:
            msg = f"Input array must be type {new_estimate.dtype} not {self.hdf5_dataset.dtype}"
            self.logger.error(msg)
            raise TypeError(msg)
        if new_estimate.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_estimate.shape)
        self.hdf5_dataset[...] = new_estimate

    def from_xarray(self, data: xr.DataArray) -> None:
        """
        Load estimate data from xarray DataArray.

        Updates metadata from xarray coordinates and attributes, then
        stores the data.

        Parameters
        ----------
        data : xr.DataArray
            DataArray containing estimate. Expected dimensions:
            (period, output, input).

        Returns
        -------
        None

        Notes
        -----
        This will update output_channels, input_channels, name, and data_type
        from the xarray object. All changes are persisted to HDF5.

        Examples
        --------
        Load estimate from modified xarray:

        >>> xr_data = est.to_xarray(periods)
        >>> # Modify data and metadata
        >>> modified = xr_data * 2  # Scale by 2
        >>> est.from_xarray(modified)
        >>> print(est.to_numpy()[0, 0, 0])  # Verify scale

        Rename channels and reload:

        >>> xr_data = est.to_xarray(periods)
        >>> new_xr = xr_data.rename({
        ...     'output': ['Ex', 'Ey'],
        ...     'input': ['Bx', 'By']
        ... })
        >>> est.from_xarray(new_xr)
        >>> print(est.metadata.output_channels)
        ['Ex', 'Ey']

        """
        self.metadata.output_channels = data.coords["output"].values.tolist()
        self.metadata.input_channels = data.coords["input"].values.tolist()
        self.metadata.name = data.name
        self.metadata.data_type = data.dtype.name

        self.write_metadata()

        self.from_numpy(data.to_numpy())
