"""
Comprehensive pytest suite for xarray helper functions.
Uses fixtures and parameterized tests for efficiency and maintainability.
"""

import numpy as np
import pytest
import xarray as xr

from mth5.timeseries.xarray_helpers import (
    covariance_xr,
    initialize_xrda_1d,
    initialize_xrda_2d,
    initialize_xrda_2d_cov,
    initialize_xrds_2d,
)


# =============================================================================
# Session-scoped fixtures for expensive operations
# =============================================================================


@pytest.fixture(scope="session")
def standard_channels():
    """Standard channel names used across tests."""
    return ["ex", "ey", "hx", "hy"]


@pytest.fixture(scope="session")
def extended_channels():
    """Extended channel names for larger tests."""
    return ["ex", "ey", "ez", "hx", "hy", "hz", "tx", "ty"]


@pytest.fixture(scope="session")
def time_array():
    """Standard time array for test data."""
    return np.arange(100, dtype=float)


@pytest.fixture(scope="session")
def frequency_array():
    """Standard frequency array for test data."""
    return np.array([0.1, 1.0, 10.0, 100.0])


@pytest.fixture(scope="session")
def long_time_array():
    """Longer time array for performance tests."""
    return np.linspace(0, 1000, 1000)


# =============================================================================
# Function-scoped fixtures for test data
# =============================================================================


@pytest.fixture
def sample_weights(time_array):
    """Random weights for weighted covariance tests."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(len(time_array))


@pytest.fixture
def real_data_matrix(time_array, standard_channels):
    """Real-valued random data matrix."""
    np.random.seed(42)
    return np.random.randn(len(time_array), len(standard_channels))


@pytest.fixture
def complex_data_matrix(time_array, standard_channels):
    """Complex-valued random data matrix."""
    np.random.seed(42)
    real_part = np.random.randn(len(time_array), len(standard_channels))
    imag_part = np.random.randn(len(time_array), len(standard_channels))
    return real_part + 1j * imag_part


@pytest.fixture
def real_xarray_data(real_data_matrix, time_array, standard_channels):
    """Real xarray DataArray with time and variable dimensions."""
    return xr.DataArray(
        real_data_matrix,
        dims=["time", "variable"],
        coords={"time": time_array, "variable": standard_channels},
    )


@pytest.fixture
def complex_xarray_data(complex_data_matrix, time_array, standard_channels):
    """Complex xarray DataArray with time and variable dimensions."""
    return xr.DataArray(
        complex_data_matrix,
        dims=["time", "variable"],
        coords={"time": time_array, "variable": standard_channels},
    )


@pytest.fixture
def sample_coords_2d(frequency_array, time_array):
    """Sample 2D coordinates dictionary for initialization tests."""
    return {"frequency": frequency_array, "time": time_array}


@pytest.fixture
def sample_coords_3d(frequency_array, time_array):
    """Sample 3D coordinates dictionary for initialization tests."""
    return {
        "time": time_array,
        "frequency": frequency_array,
    }


# =============================================================================
# Parametrized test data
# =============================================================================


@pytest.fixture(
    params=[
        (float, 0.0),
        (float, 1.5),
        (complex, 0 + 0j),
        (complex, 1 + 1j),
        (complex, 2.5 + 3.7j),
        (bool, False),
        (bool, True),
    ]
)
def dtype_value_pairs(request):
    """Parameterized dtype and value pairs for initialization tests."""
    return request.param


@pytest.fixture(
    params=[
        ["ch1"],
        ["ex", "ey"],
        ["ex", "ey", "hx", "hy"],
        ["ex", "ey", "ez", "hx", "hy", "hz"],
    ]
)
def variable_channels(request):
    """Parameterized channel lists of different sizes."""
    return request.param


# =============================================================================
# Test Classes for Organized Testing
# =============================================================================


class TestCovarianceXR:
    """Test suite for covariance_xr function."""

    def test_covariance_real_data_structure(self, real_xarray_data, standard_channels):
        """Test covariance computation structure with real data."""
        S = covariance_xr(real_xarray_data)

        # Check dimensions and coordinates
        assert S.dims == ("channel_1", "channel_2")
        assert list(S.coords["channel_1"].values) == standard_channels
        assert list(S.coords["channel_2"].values) == standard_channels
        assert S.shape == (len(standard_channels), len(standard_channels))

    def test_covariance_real_data_properties(self, real_xarray_data):
        """Test mathematical properties of real covariance matrix."""
        S = covariance_xr(real_xarray_data)

        # Check symmetry for real data
        assert np.allclose(
            S.values, S.values.T
        ), "Real covariance matrix should be symmetric"

        # Check against numpy covariance
        np_cov = np.cov(real_xarray_data.values.T)
        assert np.allclose(S.values, np_cov), "Should match numpy.cov result"

    def test_covariance_complex_data_structure(
        self, complex_xarray_data, standard_channels
    ):
        """Test covariance computation structure with complex data."""
        S = covariance_xr(complex_xarray_data)

        # Check dimensions and coordinates
        assert S.dims == ("channel_1", "channel_2")
        assert list(S.coords["channel_1"].values) == standard_channels
        assert list(S.coords["channel_2"].values) == standard_channels

    def test_covariance_complex_data_properties(self, complex_xarray_data):
        """Test mathematical properties of complex covariance matrix."""
        S = covariance_xr(complex_xarray_data)

        # Check Hermitian symmetry (conjugate transpose)
        assert np.allclose(
            S.values, S.values.conj().T
        ), "Complex covariance should be Hermitian"

        # Check against numpy covariance
        np_cov = np.cov(complex_xarray_data.values.T)
        assert np.allclose(S.values, np_cov), "Should match numpy.cov result"

    def test_covariance_with_weights_real(
        self, real_xarray_data, sample_weights, standard_channels
    ):
        """Test weighted covariance with real data."""
        S = covariance_xr(real_xarray_data, aweights=sample_weights)

        # Check structure
        assert S.dims == ("channel_1", "channel_2")
        assert list(S.coords["channel_1"].values) == standard_channels
        assert list(S.coords["channel_2"].values) == standard_channels

        # Check symmetry
        assert np.allclose(
            S.values, S.values.T
        ), "Weighted real covariance should be symmetric"

        # Check against numpy weighted covariance
        np_cov = np.cov(real_xarray_data.values.T, aweights=sample_weights)
        assert np.allclose(S.values, np_cov), "Should match numpy weighted covariance"

    def test_covariance_with_weights_complex(
        self, complex_xarray_data, sample_weights, standard_channels
    ):
        """Test weighted covariance with complex data."""
        S = covariance_xr(complex_xarray_data, aweights=sample_weights)

        # Check structure
        assert S.dims == ("channel_1", "channel_2")
        assert list(S.coords["channel_1"].values) == standard_channels
        assert list(S.coords["channel_2"].values) == standard_channels

        # Check Hermitian symmetry
        assert np.allclose(
            S.values, S.values.conj().T
        ), "Weighted complex covariance should be Hermitian"

        # Check against numpy weighted covariance
        np_cov = np.cov(complex_xarray_data.values.T, aweights=sample_weights)
        assert np.allclose(S.values, np_cov), "Should match numpy weighted covariance"

    @pytest.mark.parametrize("bias", [True, False])
    def test_covariance_bias_parameter(self, real_xarray_data, bias):
        """Test covariance with different bias settings."""
        S = covariance_xr(real_xarray_data, bias=bias)

        # Check against numpy with same bias setting
        np_cov = np.cov(real_xarray_data.values.T, bias=bias)
        assert np.allclose(S.values, np_cov), f"Should match numpy.cov with bias={bias}"

    @pytest.mark.parametrize("rowvar", [True, False])
    def test_covariance_rowvar_parameter(self, rowvar):
        """Test covariance with different rowvar settings."""
        # Create test data with appropriate orientation
        if rowvar:
            # Variables in rows, observations in columns
            data = np.random.randn(4, 100)  # 4 variables, 100 observations
            dims = ["variable", "time"]
            coords = {"variable": ["ex", "ey", "hx", "hy"], "time": np.arange(100)}
        else:
            # Variables in columns, observations in rows (default)
            data = np.random.randn(100, 4)  # 100 observations, 4 variables
            dims = ["time", "variable"]
            coords = {"time": np.arange(100), "variable": ["ex", "ey", "hx", "hy"]}

        xr_data = xr.DataArray(data, dims=dims, coords=coords)
        S = covariance_xr(xr_data, rowvar=rowvar)

        # Check against numpy with same rowvar setting
        np_cov = np.cov(xr_data.values, rowvar=rowvar)
        assert np.allclose(
            S.values, np_cov
        ), f"Should match numpy.cov with rowvar={rowvar}"

    def test_covariance_single_variable(self):
        """Test covariance with single variable (edge case)."""
        single_channel = ["ex"]
        time_data = np.arange(50)
        data = np.random.randn(50, 1)

        xr_data = xr.DataArray(
            data,
            dims=["time", "variable"],
            coords={"time": time_data, "variable": single_channel},
        )

        S = covariance_xr(xr_data)

        # Should be 1x1 matrix
        assert S.shape == (1, 1)
        assert list(S.coords["channel_1"].values) == single_channel
        assert list(S.coords["channel_2"].values) == single_channel

        # Check against numpy - the helper should convert scalar to 1x1 matrix
        np_cov = np.cov(xr_data.values.T)
        if np_cov.ndim == 0:  # numpy returns scalar for single variable
            assert np.allclose(S.values[0, 0], np_cov)
        else:  # numpy returns 1x1 array
            assert np.allclose(S.values, np_cov)


class TestInitializeXRDA1D:
    """Test suite for initialize_xrda_1d function."""

    def test_default_initialization(self, standard_channels):
        """Test default initialization (float, zero value)."""
        xrda = initialize_xrda_1d(standard_channels)

        # Check structure
        assert xrda.dims == ("variable",)
        assert list(xrda.coords["variable"].values) == standard_channels
        assert xrda.shape == (len(standard_channels),)

        # Check default values (should be zeros)
        assert np.allclose(xrda.values, np.zeros(len(standard_channels)))

    @pytest.mark.parametrize(
        "channels",
        [
            ["ch1"],
            ["ex", "ey"],
            ["ex", "ey", "hx", "hy"],
            ["ex", "ey", "ez", "hx", "hy", "hz", "tx", "ty", "tz"],
        ],
    )
    def test_variable_channel_sizes(self, channels):
        """Test initialization with different channel list sizes."""
        xrda = initialize_xrda_1d(channels)

        assert xrda.dims == ("variable",)
        assert list(xrda.coords["variable"].values) == channels
        assert xrda.shape == (len(channels),)
        assert np.allclose(xrda.values, np.zeros(len(channels)))

    def test_dtype_and_value_combinations(self, standard_channels, dtype_value_pairs):
        """Test different dtype and value combinations."""
        dtype, value = dtype_value_pairs
        xrda = initialize_xrda_1d(standard_channels, dtype=dtype, value=value)

        # Check data type
        assert xrda.dtype == dtype

        # Check values
        expected = np.full(len(standard_channels), value, dtype=dtype)
        if dtype == complex:
            assert np.allclose(xrda.values, expected)
        else:
            assert np.array_equal(xrda.values, expected)

    def test_zero_value_behavior(self, standard_channels):
        """Test that zero value doesn't trigger special assignment."""
        for dtype in [float, complex, bool]:
            xrda = initialize_xrda_1d(standard_channels, dtype=dtype, value=0)
            expected = np.zeros(len(standard_channels), dtype=dtype)
            assert np.array_equal(xrda.values, expected)

    def test_non_zero_value_assignment(self, standard_channels):
        """Test non-zero value assignment for different dtypes."""
        # Test float
        xrda_float = initialize_xrda_1d(standard_channels, dtype=float, value=3.14)
        assert np.allclose(xrda_float.values, np.full(len(standard_channels), 3.14))

        # Test complex
        xrda_complex = initialize_xrda_1d(
            standard_channels, dtype=complex, value=2 + 3j
        )
        assert np.allclose(xrda_complex.values, np.full(len(standard_channels), 2 + 3j))

        # Test bool
        xrda_bool = initialize_xrda_1d(standard_channels, dtype=bool, value=True)
        assert np.array_equal(xrda_bool.values, np.full(len(standard_channels), True))


class TestInitializeXRDS2D:
    """Test suite for initialize_xrds_2d function."""

    def test_default_initialization(self, sample_coords_2d, standard_channels):
        """Test default initialization of 2D Dataset."""
        xrds = initialize_xrds_2d(standard_channels, sample_coords_2d)

        # Check that it's a Dataset
        assert isinstance(xrds, xr.Dataset)

        # Check variables
        assert list(xrds.data_vars.keys()) == standard_channels

        # Check coordinates
        for coord_name, coord_values in sample_coords_2d.items():
            assert coord_name in xrds.coords
            assert np.array_equal(xrds.coords[coord_name].values, coord_values)

    def test_variable_properties(self, sample_coords_2d, standard_channels):
        """Test properties of individual variables in Dataset."""
        xrds = initialize_xrds_2d(standard_channels, sample_coords_2d)

        expected_dims = tuple(sample_coords_2d.keys())
        expected_shape = tuple(len(v) for v in sample_coords_2d.values())

        for var in standard_channels:
            # Check dimensions
            assert xrds[var].dims == expected_dims

            # Check shape
            assert xrds[var].shape == expected_shape

            # Check coordinates
            for coord_name, coord_values in sample_coords_2d.items():
                assert np.array_equal(xrds[var].coords[coord_name].values, coord_values)

            # Check default dtype and values
            assert np.iscomplexobj(xrds[var].values)
            assert np.all(xrds[var].values == 0)

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (complex, 0 + 0j),
            (complex, 1 + 2j),
            (float, 0.0),
            (float, 3.14),
            (bool, False),
            (bool, True),
        ],
    )
    def test_dtype_value_combinations(
        self, sample_coords_2d, standard_channels, dtype, value
    ):
        """Test different dtype and value combinations for Dataset."""
        xrds = initialize_xrds_2d(
            standard_channels, sample_coords_2d, dtype=dtype, value=value
        )

        for var in standard_channels:
            # Check dtype
            assert xrds[var].dtype == dtype

            # Check values
            if dtype in [complex, float]:
                assert np.allclose(xrds[var].values, value)
            else:
                assert np.all(xrds[var].values == value)

    def test_different_coordinate_structures(self):
        """Test Dataset initialization with different coordinate structures."""
        # Test with 1D coordinates
        coords_1d = {"time": np.arange(10)}
        variables = ["var1", "var2"]

        xrds = initialize_xrds_2d(variables, coords_1d)
        assert isinstance(xrds, xr.Dataset)
        for var in variables:
            assert xrds[var].dims == ("time",)
            assert xrds[var].shape == (10,)

        # Test with 3D coordinates
        coords_3d = {
            "time": np.arange(5),
            "frequency": [0.1, 1.0, 10.0],
            "station": ["MT001", "MT002"],
        }

        xrds = initialize_xrds_2d(variables, coords_3d)
        for var in variables:
            assert xrds[var].dims == ("time", "frequency", "station")
            assert xrds[var].shape == (5, 3, 2)

    def test_empty_variables_list(self, sample_coords_2d):
        """Test behavior with empty variables list."""
        xrds = initialize_xrds_2d([], sample_coords_2d)

        assert isinstance(xrds, xr.Dataset)
        assert len(xrds.data_vars) == 0

        # Coordinates should still be present
        for coord_name in sample_coords_2d.keys():
            assert coord_name in xrds.coords


class TestInitializeXRDA2D:
    """Test suite for initialize_xrda_2d function."""

    def test_default_initialization(self, sample_coords_3d, standard_channels):
        """Test default initialization of 2D DataArray."""
        xrda = initialize_xrda_2d(standard_channels, sample_coords_3d)

        # Check that it's a DataArray
        assert isinstance(xrda, xr.DataArray)

        # Check dimensions - should be coords dimensions plus 'variable'
        expected_dims = list(sample_coords_3d.keys()) + ["variable"]
        assert xrda.dims == tuple(expected_dims)

        # Check coordinates
        for coord_name, coord_values in sample_coords_3d.items():
            assert coord_name in xrda.coords
            assert np.array_equal(xrda.coords[coord_name].values, coord_values)

        # Check variable coordinate
        assert np.array_equal(xrda.coords["variable"].values, standard_channels)

    def test_array_properties(self, sample_coords_3d, standard_channels):
        """Test properties of the DataArray."""
        xrda = initialize_xrda_2d(standard_channels, sample_coords_3d)

        # Check shape
        expected_shape = tuple(len(v) for v in sample_coords_3d.values()) + (
            len(standard_channels),
        )
        assert xrda.shape == expected_shape

        # Check default dtype and values
        assert np.iscomplexobj(xrda.values)
        assert np.all(xrda.values == 0)

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (complex, 0 + 0j),
            (complex, 2 + 3j),
            (float, 0.0),
            (float, 1.5),
        ],
    )
    def test_dtype_value_combinations(
        self, sample_coords_3d, standard_channels, dtype, value
    ):
        """Test different dtype and value combinations."""
        xrda = initialize_xrda_2d(
            standard_channels, sample_coords_3d, dtype=dtype, value=value
        )

        # Check dtype
        assert xrda.dtype == dtype

        # Check values
        if dtype == complex:
            assert np.allclose(xrda.values, value)
        else:
            assert np.allclose(xrda.values, value)

    def test_dimension_ordering(self, standard_channels):
        """Test that dimension ordering is preserved correctly."""
        # Test with explicit coordinate ordering
        coords = {
            "frequency": [0.1, 1.0, 10.0],
            "time": np.arange(50),
        }

        xrda = initialize_xrda_2d(standard_channels, coords)

        # Dimensions should follow the order: coord keys + 'variable'
        expected_dims = ("frequency", "time", "variable")
        assert xrda.dims == expected_dims

        expected_shape = (3, 50, len(standard_channels))
        assert xrda.shape == expected_shape

    def test_conversion_from_dataset(self, sample_coords_3d, standard_channels):
        """Test that conversion from Dataset maintains data correctly."""
        # Create Dataset first
        xrds = initialize_xrds_2d(standard_channels, sample_coords_3d, value=2 + 3j)

        # Convert to DataArray manually
        xrda_manual = xrds.to_array(dim="variable")
        dims = list(sample_coords_3d.keys())
        xrda_manual = xrda_manual.transpose(*dims, "variable")

        # Create using initialize_xrda_2d
        xrda_init = initialize_xrda_2d(
            standard_channels, sample_coords_3d, value=2 + 3j
        )

        # Should be equivalent
        assert xrda_manual.dims == xrda_init.dims
        assert xrda_manual.shape == xrda_init.shape
        assert np.array_equal(
            xrda_manual.coords["variable"].values, xrda_init.coords["variable"].values
        )
        assert np.allclose(xrda_manual.values, xrda_init.values)


class TestInitializeXRDA2DCov:
    """Test suite for initialize_xrda_2d_cov function."""

    def test_default_initialization(self, standard_channels):
        """Test default initialization of covariance DataArray."""
        xrda = initialize_xrda_2d_cov(standard_channels)

        # Check dimensions
        assert xrda.dims == ("channel_1", "channel_2")

        # Check coordinates
        assert list(xrda.coords["channel_1"].values) == standard_channels
        assert list(xrda.coords["channel_2"].values) == standard_channels

        # Check shape (should be square)
        expected_shape = (len(standard_channels), len(standard_channels))
        assert xrda.shape == expected_shape

        # Check default dtype and values
        assert xrda.dtype == complex
        assert np.allclose(xrda.values, np.zeros(expected_shape))

    @pytest.mark.parametrize(
        "channels",
        [
            ["ch1"],
            ["ex", "ey"],
            ["ex", "ey", "hx", "hy"],
            ["ex", "ey", "ez", "hx", "hy", "hz", "tx", "ty"],
        ],
    )
    def test_variable_channel_sizes(self, channels):
        """Test covariance matrix with different channel sizes."""
        xrda = initialize_xrda_2d_cov(channels)

        expected_shape = (len(channels), len(channels))
        assert xrda.shape == expected_shape
        assert list(xrda.coords["channel_1"].values) == channels
        assert list(xrda.coords["channel_2"].values) == channels

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (complex, 0 + 0j),
            (complex, 1 + 2j),
            (complex, -1.5 - 2.7j),
            (float, 0.0),
            (float, 3.14),
            (float, -2.5),
        ],
    )
    def test_dtype_value_combinations(self, standard_channels, dtype, value):
        """Test different dtype and value combinations for covariance matrix."""
        xrda = initialize_xrda_2d_cov(standard_channels, dtype=dtype, value=value)

        # Check dtype
        assert xrda.dtype == dtype

        # Check values
        expected = np.full(
            (len(standard_channels), len(standard_channels)), value, dtype=dtype
        )
        if dtype == complex:
            assert np.allclose(xrda.values, expected)
        else:
            assert np.allclose(xrda.values, expected)

    def test_zero_value_behavior(self, standard_channels):
        """Test that zero value doesn't trigger special assignment."""
        for dtype in [complex, float]:
            xrda = initialize_xrda_2d_cov(standard_channels, dtype=dtype, value=0)
            expected_shape = (len(standard_channels), len(standard_channels))
            expected = np.zeros(expected_shape, dtype=dtype)
            assert np.array_equal(xrda.values, expected)

    def test_non_zero_value_assignment(self, standard_channels):
        """Test non-zero value assignment."""
        # Test complex
        xrda_complex = initialize_xrda_2d_cov(
            standard_channels, dtype=complex, value=2 + 3j
        )
        expected_complex = np.full(
            (len(standard_channels), len(standard_channels)), 2 + 3j, dtype=complex
        )
        assert np.allclose(xrda_complex.values, expected_complex)

        # Test float
        xrda_float = initialize_xrda_2d_cov(standard_channels, dtype=float, value=1.5)
        expected_float = np.full(
            (len(standard_channels), len(standard_channels)), 1.5, dtype=float
        )
        assert np.allclose(xrda_float.values, expected_float)

    def test_single_channel_covariance(self):
        """Test covariance matrix with single channel (edge case)."""
        single_channel = ["ex"]
        xrda = initialize_xrda_2d_cov(single_channel)

        assert xrda.shape == (1, 1)
        assert list(xrda.coords["channel_1"].values) == single_channel
        assert list(xrda.coords["channel_2"].values) == single_channel


# =============================================================================
# Integration and Performance Tests
# =============================================================================


class TestIntegrationAndPerformance:
    """Integration tests and performance validations."""

    def test_covariance_initialization_integration(
        self, standard_channels, real_xarray_data
    ):
        """Test integration between covariance computation and initialization."""
        # Compute covariance
        S = covariance_xr(real_xarray_data)

        # Create initialized covariance matrix
        S_init = initialize_xrda_2d_cov(standard_channels, dtype=complex)

        # Should have same structure
        assert S.dims == S_init.dims
        assert S.shape == S_init.shape
        assert list(S.coords["channel_1"].values) == list(
            S_init.coords["channel_1"].values
        )
        assert list(S.coords["channel_2"].values) == list(
            S_init.coords["channel_2"].values
        )

    def test_dataset_dataarray_conversion_consistency(
        self, sample_coords_3d, standard_channels
    ):
        """Test consistency between Dataset and DataArray initialization."""
        # Initialize Dataset
        xrds = initialize_xrds_2d(standard_channels, sample_coords_3d, value=1 + 2j)

        # Initialize DataArray
        xrda = initialize_xrda_2d(standard_channels, sample_coords_3d, value=1 + 2j)

        # Convert Dataset to DataArray
        xrda_from_ds = xrds.to_array(dim="variable")
        dims = list(sample_coords_3d.keys())
        xrda_from_ds = xrda_from_ds.transpose(*dims, "variable")

        # Should be equivalent
        assert xrda.dims == xrda_from_ds.dims
        assert xrda.shape == xrda_from_ds.shape
        assert np.allclose(xrda.values, xrda_from_ds.values)

    def test_large_array_performance(self, extended_channels, long_time_array):
        """Test performance with larger arrays."""
        # Create large real data
        np.random.seed(42)
        large_data = np.random.randn(len(long_time_array), len(extended_channels))
        large_xr = xr.DataArray(
            large_data,
            dims=["time", "variable"],
            coords={"time": long_time_array, "variable": extended_channels},
        )

        # Test covariance computation
        S = covariance_xr(large_xr)
        assert S.shape == (len(extended_channels), len(extended_channels))
        assert np.allclose(S.values, S.values.T)  # Symmetry check

        # Test large initialization
        large_coords = {"time": long_time_array, "frequency": np.logspace(-2, 2, 50)}
        xrds = initialize_xrds_2d(extended_channels, large_coords)
        assert len(xrds.data_vars) == len(extended_channels)

        for var in extended_channels:
            assert xrds[var].shape == (len(long_time_array), 50)

    @pytest.mark.parametrize(
        "n_channels", [2, 4, 8, 16]
    )  # Skip n_channels=1 to avoid scalar covariance issue
    def test_scaling_with_channel_count(self, n_channels, time_array):
        """Test how functions scale with number of channels."""
        channels = [f"ch{i}" for i in range(n_channels)]

        # Test initialization scaling
        xrda_1d = initialize_xrda_1d(channels)
        assert xrda_1d.shape == (n_channels,)

        xrda_cov = initialize_xrda_2d_cov(channels)
        assert xrda_cov.shape == (n_channels, n_channels)

        # Test covariance scaling (skip for n_channels=1 due to numpy scalar return)
        data = np.random.randn(len(time_array), n_channels)
        xr_data = xr.DataArray(
            data,
            dims=["time", "variable"],
            coords={"time": time_array, "variable": channels},
        )

        S = covariance_xr(xr_data)
        assert S.shape == (n_channels, n_channels)

    def test_single_channel_scaling(self, time_array):
        """Test single channel case separately due to numpy scalar covariance behavior."""
        channels = ["ch0"]

        # Test initialization scaling
        xrda_1d = initialize_xrda_1d(channels)
        assert xrda_1d.shape == (1,)

        xrda_cov = initialize_xrda_2d_cov(channels)
        assert xrda_cov.shape == (1, 1)

        # Test covariance - the helper should handle numpy scalar return for single channel
        data = np.random.randn(len(time_array), 1)
        xr_data = xr.DataArray(
            data,
            dims=["time", "variable"],
            coords={"time": time_array, "variable": channels},
        )

        S = covariance_xr(xr_data)
        assert S.shape == (1, 1)

        # Verify against numpy (which returns scalar for single variable)
        np_cov = np.cov(xr_data.values.T)
        # The helper should convert scalar to 1x1 matrix
        assert np.allclose(S.values[0, 0], np_cov)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_empty_channel_list_handling(self):
        """Test behavior with empty channel lists."""
        # Most functions should handle empty lists gracefully
        empty_channels = []

        # initialize_xrda_1d with empty list
        xrda = initialize_xrda_1d(empty_channels)
        assert xrda.shape == (0,)
        assert len(xrda.coords["variable"]) == 0

        # initialize_xrda_2d_cov with empty list
        xrda_cov = initialize_xrda_2d_cov(empty_channels)
        assert xrda_cov.shape == (0, 0)

    def test_single_time_point_covariance(self, standard_channels):
        """Test covariance with single time point (edge case)."""
        # Single time point will result in NaN covariance (degrees of freedom issue)
        single_time = np.array([0.0])
        data = np.random.randn(1, len(standard_channels))

        xr_data = xr.DataArray(
            data,
            dims=["time", "variable"],
            coords={"time": single_time, "variable": standard_channels},
        )

        # covariance_xr returns all-NaN for < 2 time samples without raising warnings
        S = covariance_xr(xr_data)

        assert S.shape == (len(standard_channels), len(standard_channels))
        # With single point, covariance will be NaN due to degrees of freedom
        assert np.all(np.isnan(S.values))

        # Verify this matches numpy behavior
        with pytest.warns(RuntimeWarning):
            np_cov = np.cov(xr_data.values.T)
        assert np.all(np.isnan(np_cov))

    def test_coordinates_with_different_dtypes(self):
        """Test initialization with different coordinate data types."""
        channels = ["ex", "ey"]

        # Mixed coordinate types
        coords = {
            "time": np.arange(10, dtype=int),
            "frequency": np.array([1.0, 10.0], dtype=float),
            "station": ["MT001", "MT002"],  # string coordinates
        }

        xrds = initialize_xrds_2d(channels, coords)

        for var in channels:
            assert xrds[var].dims == ("time", "frequency", "station")
            # Check that coordinates are preserved with correct types
            assert xrds[var].coords["time"].dtype == int
            assert xrds[var].coords["frequency"].dtype == float
            assert xrds[var].coords["station"].dtype.kind in ["U", "O"]  # string types

    def test_very_small_arrays(self):
        """Test with minimal array sizes."""
        single_channel = ["ch1"]
        single_time = np.array([0.0])

        # Test 1D initialization
        xrda_1d = initialize_xrda_1d(single_channel)
        assert xrda_1d.shape == (1,)

        # Test covariance matrix
        xrda_cov = initialize_xrda_2d_cov(single_channel)
        assert xrda_cov.shape == (1, 1)

        # Test 2D Dataset
        coords = {"time": single_time}
        xrds = initialize_xrds_2d(single_channel, coords)
        assert xrds[single_channel[0]].shape == (1,)


# =============================================================================
# Module Import Test
# =============================================================================


def test_module_imports():
    """Test that all required functions can be imported."""
    from mth5.timeseries.xarray_helpers import (
        covariance_xr,
        initialize_xrda_1d,
        initialize_xrda_2d,
        initialize_xrda_2d_cov,
        initialize_xrds_2d,
    )

    # Test that functions are callable
    assert callable(covariance_xr)
    assert callable(initialize_xrda_1d)
    assert callable(initialize_xrda_2d)
    assert callable(initialize_xrda_2d_cov)
    assert callable(initialize_xrds_2d)
