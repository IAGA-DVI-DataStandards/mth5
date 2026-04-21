"""
Module containing helper functions for working with xarray objects.
"""

from typing import Optional, Union

import numpy as np
import xarray as xr


def covariance_xr(
    X: xr.DataArray,
    aweights: Optional[Union[np.ndarray, None]] = None,
    bias: Optional[bool] = False,
    rowvar: Optional[bool] = False,
) -> xr.DataArray:
    """
    Compute the covariance matrix with numpy.cov.

    Parameters
    ----------
    X: xarray.core.dataarray.DataArray
        Multivariate time series as an xarray
    aweights: array_like, optional
        Passthrough param for np.cov.
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.
    bias: bool
        Passthrough param for np.cov.
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    rowvar: bool
        Passthrough param for np.cov.
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.

    Returns
    -------
    S: xarray.DataArray
        The covariance matrix of the data in xarray form.

    Development Notes:
        In case of ValueError: conflicting sizes for dimension 'channel_1', this likely means the bool for rowvar
        should be flipped.
    """

    channels = list(X.coords["variable"].values)

    # Guard against divide-by-zero when fewer than 2 time samples are present
    if X.shape[0] < 2:
        nan_matrix = np.full((len(channels), len(channels)), np.nan)
        return xr.DataArray(
            nan_matrix,
            dims=["channel_1", "channel_2"],
            coords={"channel_1": channels, "channel_2": channels},
        )

    cov_result = np.cov(X.values, rowvar=rowvar, aweights=aweights, bias=bias)

    # Handle scalar result for single variable
    if cov_result.ndim == 0:
        cov_result = np.array([[cov_result]])

    S = xr.DataArray(
        cov_result,
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    return S


def initialize_xrda_1d(
    channels: list,
    dtype: Optional[type] = None,
    value: Optional[Union[complex, float, bool]] = 0.0,
) -> xr.DataArray:
    """
    Returns a 1D xr.DataArray with variable "channel", having values channels named by the input list.

    Parameters
    ----------
    channels: list
        The channels in the multivariate array
    dtype: type, optional
        The datatype to initialize the array.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool], optional
        The default value to assign the array

    Returns
    -------
    xrda: xarray.core.dataarray.DataArray
        An xarray container for the channels, initialized to zeros.
    """
    k = len(channels)
    xrda = xr.DataArray(
        np.zeros(k, dtype=dtype),
        dims=[
            "variable",
        ],
        coords={
            "variable": channels,
        },
    )
    if value != 0:
        data = value * np.ones(k, dtype=dtype)
        xrda.data = data
    return xrda


def initialize_xrds_2d(
    variables: list,
    coords: dict,
    dtype: Optional[type] = complex,
    value: Optional[Union[complex, float, bool]] = 0,
) -> xr.Dataset:
    """
    Returns a 2D xr.Dataset with the given variables and coordinates.

    Parameters
    ----------
    variables: list
        List of variable names to create in the dataset
    coords: dict
        Dictionary of coordinates for the dataset dimensions
    dtype: type, optional
        The datatype to initialize the arrays.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool], optional
        The default value to assign the arrays

    Returns
    -------
    xrds: xr.Dataset
        A 2D xarray Dataset with dimensions from coords
    """
    # Get dimensions from coords
    dims = list(coords.keys())
    shape = tuple(len(v) for v in coords.values())

    # Initialize empty dataset
    xrds = xr.Dataset(coords=coords)

    # Add each variable
    for var in variables:
        if value == 0:
            data = np.zeros(shape, dtype=dtype)
        else:
            data = value * np.ones(shape, dtype=dtype)

        xrds[var] = xr.DataArray(data, dims=dims, coords=coords)

    return xrds


def initialize_xrda_2d(
    variables: list, coords: dict, dtype: type = complex, value: Union[int, float] = 0.0
) -> xr.Dataset:
    """Initialize a 3D xarray DataArray with dimensions from coords plus 'variable'.

    Parameters
    ----------
    variables : list
        List of variable names for the additional dimension.
    coords : dict
        Dictionary of coordinates for the dataset dimensions.
    dtype : type, optional
        Data type for the array, by default complex.
    value : int or float, optional
        Value to initialize the array with, by default 0.

    Returns
    -------
    xr.DataArray
        A 3D DataArray with dimensions from coords plus 'variable'.
    """
    # Create Dataset first
    ds = initialize_xrds_2d(variables, coords, dtype, value)

    # Convert to DataArray with original dimension order plus 'variable'
    dims = list(coords.keys())
    da = ds.to_array(dim="variable").transpose(*dims, "variable")

    return da


def initialize_xrda_2d_cov(
    channels, dtype=complex, value: Optional[Union[complex, float, bool]] = 0
):
    """
     TODO: consider changing nomenclature from dims=["channel_1", "channel_2"],
     to dims=["variable_1", "variable_2"], to be consistent with initialize_xrda_1d

    Parameters
     ----------
     channels: list
         The channels in the multivariate array. The covariance matrix will be square
         with dimensions len(channels) x len(channels).
     dtype: type
         The datatype to initialize the array.
         Common cases are complex, float, and bool
     value: Union[complex, float, bool]
         The default value to assign the array

    Returns
     -------
     xrda: xarray.core.dataarray.DataArray
         An xarray container for the channel covariances, initialized to zeros.
         The matrix is square with dimensions len(channels) x len(channels).
    """
    K = len(channels)
    xrda = xr.DataArray(
        np.zeros((K, K), dtype=dtype),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    if value != 0:
        data = value * np.ones(xrda.shape, dtype=dtype)
        xrda.data = data

    return xrda
