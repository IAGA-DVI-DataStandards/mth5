"""Top-level package for MTH5."""

# =============================================================================
# Imports
# =============================================================================
import sys
from importlib import import_module
import numpy as np
import xarray as xr
import h5py
from loguru import logger

try:
    from mt_io.reader import read_file
except ModuleNotFoundError:

    def read_file(*args, **kwargs):
        raise ModuleNotFoundError(
            "mt_io is required for read_file(). Install mt-io to enable file readers."
        )


import mt_timeseries.scipy_filters

# # Register xarray accessors
# import mt_timeseries.channel_dataset_accessor

# =============================================================================
# Package Variables
# =============================================================================

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.6.8"


# =============================================================================
# Initialize Loggers
# =============================================================================
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "level": "INFO",
            "colorize": True,
            "format": "<level>{time} | {level: <3} | {name} | {function} | line: {line} | {message}</level>",
        },
    ],
    "extra": {"user": "someone"},
}
logger.configure(**config)
# logger.disable("mth5")

# need to set this to make sure attributes of data arrays and data sets
# are kept when doing xarray computations like merge.
xr.set_options(keep_attrs=True)

# =============================================================================
# Defualt Parameters
# =============================================================================
CHUNK_SIZE = 8196
ACCEPTABLE_FILE_TYPES = ["mth5", "MTH5", "h5", "H5"]
ACCEPTABLE_FILE_SUFFIXES = [f".{x}" for x in ACCEPTABLE_FILE_TYPES]
ACCEPTABLE_FILE_VERSIONS = ["0.1.0", "0.2.0"]
ACCEPTABLE_DATA_LEVELS = [0, 1, 2, 3]

### transfer function summary table dtype
TF_DTYPE_LIST = [
    ("station", "S30"),
    ("survey", "S50"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("tf_id", "S30"),
    ("units", "S60"),
    ("has_impedance", bool),
    ("has_tipper", bool),
    ("has_covariance", bool),
    ("period_min", float),
    ("period_max", float),
    ("hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

TF_DTYPE = np.dtype(TF_DTYPE_LIST)

### Channel summary table dtype
CHANNEL_DTYPE_LIST = [
    ("survey", "S30"),
    ("station", "S30"),
    ("run", "S20"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("component", "S20"),
    ("start", "S36"),
    ("end", "S36"),
    ("n_samples", np.int64),
    ("sample_rate", float),
    ("measurement_type", "S30"),
    ("azimuth", float),
    ("tilt", float),
    ("units", "S60"),
    ("has_data", bool),
    ("hdf5_reference", h5py.ref_dtype),
    ("run_hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

CHANNEL_DTYPE = np.dtype(CHANNEL_DTYPE_LIST)

### Fourier coefficient summary table dtype
FC_DTYPE_LIST = [
    ("survey", "S30"),
    ("station", "S30"),
    ("run", "S20"),
    ("decimation_level", "S44"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("component", "S20"),
    ("start", "S36"),
    ("end", "S36"),
    ("n_samples", np.int64),
    ("sample_rate", float),
    ("measurement_type", "S30"),
    ("units", "S60"),
    ("hdf5_reference", h5py.ref_dtype),
    ("decimation_level_reference", h5py.ref_dtype),
    ("run_hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

FC_DTYPE = np.dtype(FC_DTYPE_LIST)
### Run summary table dtype

RUN_SUMMARY_LIST = [
    ("channel_scale_factors", float),
    ("duration", float),
    ("end", str),
    ("has_data", bool),
    ("input_channels", list),
    ("mth5_path", str),
    ("n_samples", int),
    ("output_channels", list),
    ("run", str),
    ("sample_rate", float),
    ("start", str),
    ("station", str),
    ("survey", str),
    ("run_hdf5_reference", object),
    ("station_hdf5_reference", object),
]

RUN_SUMMARY_DTYPE = np.dtype(RUN_SUMMARY_LIST)

RUN_SUMMARY_COLUMNS = [entry[0] for entry in RUN_SUMMARY_LIST]

### Standards dtype
STANDARDS_DTYPE_LIST = [
    ("attribute", "S72"),
    ("type", "S15"),
    ("required", bool),
    ("style", "S72"),
    ("units", "S32"),
    ("description", "S300"),
    ("options", "S150"),
    ("alias", "S72"),
    ("example", "S72"),
    ("default", "S72"),
]
STANDARDS_DTYPE = np.dtype(STANDARDS_DTYPE_LIST)


_LAZY_SUBMODULES = {"clients", "io"}


def __getattr__(name):
    """Lazily expose package submodules for dotted-path resolvers."""
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
