# -*- coding: utf-8 -*-
"""
Universal reader for magnetotelluric time series data files.

This module provides a plugin-like system for reading various MT data formats
and returning appropriate :class:`mth5.timeseries` objects. The reader
automatically detects file types and dispatches to the correct parser.

Plugin Structure
----------------
If you are writing your own reader, implement the following structure:

    * Class object that reads the given file format
    * A reader function named read_{file_type} (e.g., read_nims)
    * Return value should be a :class:`mth5.timeseries.MTTS` or
      :class:`mth5.timeseries.RunTS` object plus extra metadata as a
      dictionary with keys formatted as {level.attribute}

Example Implementation
----------------------
.. code-block:: python

    class NewFile:
        def __init__(self, fn):
            self.fn = fn

        def read_header(self):
            return header_information

        def read_newfile(self):
            ex, ey, hx, hy, hz = read_in_channels_as_MTTS
            return RunTS([ex, ey, hx, hy, hz])

    def read_newfile(fn):
        new_file_obj = NewFile(fn)
        run_obj = new_file_obj.read_newfile()
        return run_obj, extra_metadata

Then add your reader to the readers dictionary for automatic detection.

See Also
--------
Existing readers in `mth5.io` for implementation guidance.

Created on Wed Aug 26 10:32:45 2020

:author: Jared Peacock
:license: MIT
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from mth5.io import lemi, lemi417, metronix, miniseed, nims, phoenix, usgs_ascii, zen


# =============================================================================
# Reader registry for MT data formats
# =============================================================================
readers: dict[str, dict[str, Any]] = {
    "zen": {"file_types": ["z3d"], "reader": zen.read_z3d},
    "nims": {"file_types": ["bin", "bnn"], "reader": nims.read_nims},
    "usgs_ascii": {
        "file_types": ["asc", "zip"],
        "reader": usgs_ascii.read_ascii,
    },
    "miniseed": {
        "file_types": ["miniseed", "ms", "mseed"],
        "reader": miniseed.read_miniseed,
    },
    "lemi424": {
        "file_types": ["txt"],
        "reader": lemi.read_lemi424,
    },
    "lemi417": {
        "file_types": ["b*"],
        "reader": lemi417.read_lemi417,
    },
    "phoenix": {
        "file_types": ["bin", "td_30", "td_150", "td_24k"],
        "reader": phoenix.read_phoenix,
    },
    "metronix": {
        "file_types": ["atss"],
        "reader": metronix.read_atss,
    },
}


def get_reader(extension: str) -> tuple[str, Callable]:
    """
    Get the appropriate reader function for a file extension.

    Searches the reader registry to find the correct parser function
    for the given file extension. Handles ambiguous extensions by
    issuing warnings when multiple readers might apply.

    Parameters
    ----------
    extension : str
        File extension (without the dot) to find a reader for

    Returns
    -------
    tuple[str, Callable]
        Tuple containing:
        - Reader name (str): Identifier for the reader type
        - Reader function (Callable): Function to parse files of this type

    Raises
    ------
    ValueError
        If no reader is found for the given file extension

    Examples
    --------
    >>> reader_name, reader_func = get_reader("z3d")
    >>> print(reader_name)  # "zen"
    >>> data = reader_func("/path/to/file.z3d")

    Notes
    -----
    Some extensions like "bin" are ambiguous and could match multiple
    readers (NIMS or Phoenix). A warning is issued in such cases.
    """
    if extension in ["bin"]:
        logger.warning("Suggest inputing file type, bin could be nims or phoenix")
    for key, vdict in readers.items():
        for file_type in vdict["file_types"]:
            if fnmatch.fnmatch(extension.lower(), file_type.lower()):
                return key, vdict["reader"]
    msg = f"Could not find a reader for file type {extension}"
    logger.error(msg)
    raise ValueError(msg)


def read_file(
    fn: str | Path | list[str | Path], file_type: str | None = None, **kwargs: Any
) -> Any:
    """
    Universal reader for magnetotelluric time series data files.

    Automatically detects the file type based on extension and dispatches
    to the appropriate reader function. Supports both single files and
    lists of files for multi-file formats.

    Parameters
    ----------
    fn : str, Path, or list of str/Path
        Full path(s) to data file(s) to be read. For multi-file formats,
        pass a list of file paths.
    file_type : str, optional
        Specific reader type to use if file extension is ambiguous.
        Must be one of the keys in the readers registry, by default None
    **kwargs : dict
        Additional keyword arguments passed to the specific reader function.
        Supported arguments depend on the file format and reader.

    Returns
    -------
    MTTS or RunTS
        Time series object containing the data:
        - :class:`mth5.timeseries.MTTS` for single channel data
        - :class:`mth5.timeseries.RunTS` for multi-channel run data

    Raises
    ------
    IOError
        If any specified file does not exist
    KeyError
        If the specified file_type is not supported
    ValueError
        If no reader can be found for the file extension

    Examples
    --------
    Read a single Z3D file (auto-detected)

    >>> data = read_file("/path/to/station_001.z3d")
    >>> print(type(data))  # <class 'mth5.timeseries.ChannelTS'>

    Read with explicit file type for ambiguous extensions

    >>> data = read_file("/path/to/data.bin", file_type="nims")
    >>> print(data.n_channels)

    Read multiple files for a multi-file format

    >>> files = ["/path/to/file1.asc", "/path/to/file2.asc"]
    >>> run_data = read_file(files, sample_rate=1.0)

    Notes
    -----
    Supported file types and extensions:
    - zen: .z3d (Zonge Z3D files)
    - nims: .bin, .bnn (USGS NIMS files)
    - usgs_ascii: .asc, .zip (USGS ASCII format)
    - miniseed: .miniseed, .ms, .mseed (miniSEED format)
    - lemi424: .txt (LEMI-424 format)
    - lemi417: .B* (such as:.B16, .B21, LEMI-417 format)
    - phoenix: .bin, .td_30, .td_150, .td_24k (Phoenix formats)
    - metronix: .atss (Metronix ADU format)

    For ambiguous extensions like .bin, specify file_type explicitly.
    """

    if isinstance(fn, (list, tuple)):
        fn = [Path(ff) for ff in fn]
        if not fn[0].exists():
            msg = f"Could not find file {fn}. Check path."
            logger.error(msg)
            raise IOError(msg)
        file_ext = fn[0].suffix[1:]

    else:
        fn = Path(fn)
        if not fn.exists():
            msg = f"Could not find file {fn}. Check path."
            logger.error(msg)
            raise IOError(msg)
        file_ext = fn.suffix[1:]

    if file_type is not None:
        try:
            file_reader = readers[file_type]["reader"]
        except KeyError:
            msg = (
                f"Reader for the file type {file_type} does not currently exist. "
                f"Current readers {list(readers.keys())}"
            )
            logger.error(msg)
            raise KeyError(msg)
    else:
        file_type, file_reader = get_reader(file_ext)
    return file_reader(fn, **kwargs)
