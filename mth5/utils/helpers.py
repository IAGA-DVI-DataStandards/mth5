# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import functools
import pathlib
from typing import Any, Callable, TypeVar

from loguru import logger

from mth5.helpers import close_open_files
from mth5.mth5 import MTH5


# =============================================================================
# Module Documentation
# =============================================================================
"""
MTH5 Utility Helper Functions.

Provides decorators and utility functions for working with MTH5 objects,
including path/object conversion, file operations, and data validation.

Notes
-----
Many functions use the `path_or_mth5_object` decorator to transparently
handle both file paths and MTH5 objects as input.

Examples
--------
Initialize and open an MTH5 file::

    >>> m = initialize_mth5('/path/to/file.mth5', mode='a')
    >>> m.close_mth5()
"""

T = TypeVar("T")


# =============================================================================


def path_or_mth5_object(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator allowing functions to accept MTH5 file paths or MTH5 objects.

    Transparently converts file paths to MTH5 objects, opens the file,
    and passes the MTH5 object to the decorated function.

    Parameters
    ----------
    func : Callable
        A function that takes an MTH5 object as its first argument.
        Signature: func(mth5_obj: MTH5, *args, **kwargs) -> T

    Returns
    -------
    Callable
        Wrapped function accepting str/Path or MTH5 as first argument.

    Raises
    ------
    TypeError
        If first argument is not a string, pathlib.Path, or MTH5 object.

    Notes
    -----
    The decorated function can be called with either:
    - A file path string or pathlib.Path
    - An MTH5 object

    When given a file path, the decorator automatically opens the file
    in 'append' mode by default, unless overridden in kwargs.

    TODO: add support for file_version in kwargs

    Examples
    --------
    Decorate a function to work with both paths and objects::

        @path_or_mth5_object
        def get_metadata(m: MTH5) -> dict:
            return m.survey_group.metadata.to_dict()

        # Call with file path
        metadata = get_metadata('/path/to/file.mth5')

        # Call with MTH5 object
        with MTH5() as m:
            m.open_mth5('/path/to/file.mth5', mode='r')
            metadata = get_metadata(m)
    """

    @functools.wraps(func)
    def wrapper_decorator(*args: Any, **kwargs: Any) -> T:
        def call_function(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
            if isinstance(func, staticmethod):
                callable_func = func.__get__(None, object)
                result = callable_func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        if isinstance(args[0], (pathlib.Path, str)):
            h5_path = args[0]
            mode = kwargs.get("mode", "a")
            # with MTH5().open_mth5(h5_path, mode=mode) as m:
            with MTH5() as m:
                m.open_mth5(h5_path, mode=mode)
                new_args = [x for x in args]
                new_args[0] = m
                new_args = tuple(new_args)
                result = call_function(func, *new_args, **kwargs)

        elif isinstance(args[0], MTH5):
            result = call_function(func, *args, **kwargs)
        else:
            msg = f"expected h5, got {type(args[0])}"
            logger.error(msg)
            raise TypeError(msg)

        return result

    return wrapper_decorator  # type: ignore


@path_or_mth5_object
def get_version(m: str | pathlib.Path | MTH5) -> str:
    """
    Get the file version from an MTH5 file.

    Parameters
    ----------
    m : str | pathlib.Path | MTH5
        Path to MTH5 file or MTH5 object.

    Returns
    -------
    str
        File version string (e.g., '0.1.0', '0.2.0').

    Examples
    --------
    Get version from file path::

        >>> version = get_version('/path/to/file.mth5')
        >>> print(version)
        '0.2.0'

    Get version from MTH5 object::

        >>> with MTH5() as m:
        ...     m.open_mth5('/path/to/file.mth5')
        ...     version = get_version(m)
    """
    return m.file_version  # type: ignore


@path_or_mth5_object
def get_channel_summary(m: str | pathlib.Path | MTH5, show: bool = True) -> Any:
    """
    Get channel summary from MTH5 file as pandas DataFrame.

    Retrieves the channel summary table and converts to DataFrame.
    Automatically re-summarizes if the summary appears incomplete.

    Parameters
    ----------
    m : str | pathlib.Path | MTH5
        Path to MTH5 file or MTH5 object.
    show : bool, default True
        Whether to log the summary DataFrame to console.

    Returns
    -------
    pandas.DataFrame
        Channel summary with station, run, and channel information.

    Warnings
    --------
    If the summary appears incomplete, the channel summary table is
    re-summarized which may take time for large files.

    Examples
    --------
    Get channel summary from file path::

        >>> df = get_channel_summary('/path/to/file.mth5')
        >>> print(df.shape)
        (42, 8)

    Get summary without logging::

        >>> df = get_channel_summary('/path/to/file.mth5', show=False)
    """
    logger.info(f"{m.filename} channel summary")  # type: ignore
    df = m.channel_summary.to_dataframe()  # type: ignore
    if len(df) <= 1:
        logger.warning("channel summary smaller than expected -- re-summarizing")
        m.channel_summary.summarize()  # type: ignore
        df = m.channel_summary.to_dataframe()  # type: ignore
    if show:
        logger.info(f"{df}")
    return df


@path_or_mth5_object
def add_filters(
    m: str | pathlib.Path | MTH5,
    filters_list: list[Any],
    survey_id: str = "",
) -> None:
    """
    Add filter objects to MTH5 file.

    Adds a list of filter objects to the MTH5 file's filter group.
    Automatically selects the appropriate filters group based on file version.

    Parameters
    ----------
    m : str | pathlib.Path | MTH5
        Path to MTH5 file or MTH5 object.
    filters_list : list
        List of filter objects to add. Each filter should have a 'name'
        attribute and be compatible with the filters group.
    survey_id : str, default ''
        Survey ID for file version 0.2.0. Required for version 0.2.0,
        ignored for version 0.1.0.

    Raises
    ------
    AttributeError
        If filter objects lack required attributes.
    ValueError
        If survey_id is not found in version 0.2.0 files.

    Notes
    -----
    File version 0.1.0 stores filters globally.
    File version 0.2.0 stores filters per survey.

    Examples
    --------
    Add filters to MTH5 file::

        >>> from mt_timeseries import Filter
        >>> filters = [Filter(name='test_filter')]
        >>> add_filters('/path/to/file.mth5', filters)

    Add survey-specific filters (version 0.2.0)::

        >>> add_filters('/path/to/file.mth5', filters, survey_id='MT01')
    """
    if m.file_version == "0.1.0":  # type: ignore
        fg = m.filters_group  # type: ignore
        assert fg is not None
    else:
        # m.file_version == "0.2.0":
        survey = m.get_survey(survey_id)  # type: ignore
        fg = survey.filters_group

    for filt3r in filters_list:
        if filt3r.name not in fg.filter_dict.keys():  # type: ignore
            fg.add_filter(filt3r)  # type: ignore
    return


def initialize_mth5(
    h5_path: str | pathlib.Path,
    mode: str = "a",
    file_version: str = "0.1.0",
) -> MTH5:
    """
    Initialize and open an MTH5 file for reading or writing.

    Creates or opens an MTH5 file with specified file version.
    Optionally removes existing files before write operations.

    Parameters
    ----------
    h5_path : str | pathlib.Path
        Path to MTH5 file. Created if it doesn't exist.
    mode : {'r', 'w', 'a'}, default 'a'
        File access mode:
        - 'r': read-only
        - 'w': write (overwrites existing file)
        - 'a': append/read-write
    file_version : {'0.1.0', '0.2.0'}, default '0.1.0'
        MTH5 file format version.

    Returns
    -------
    MTH5
        Initialized and opened MTH5 object.

    Warnings
    --------
    When mode='w' and file exists, all open h5 files are closed before
    removal. This may affect other processes using HDF5 files.

    Examples
    --------
    Create a new MTH5 file::

        >>> m = initialize_mth5('/path/to/file.mth5', mode='w')
        >>> m.file_version
        '0.1.0'
        >>> m.close_mth5()

    Open existing file for appending::

        >>> m = initialize_mth5('/path/to/file.mth5', mode='a')
        >>> m.add_station('MT001')
        >>> m.close_mth5()

    Open file with version 0.2.0 schema::

        >>> m = initialize_mth5('/path/to/file.mth5', file_version='0.2.0')
    """
    h5_path = pathlib.Path(h5_path)
    if mode == "w":
        if h5_path.exists():
            msg = f"File {h5_path} exists, removing from file system."
            msg = f"{msg}\n closing all open h5 files before removal"
            logger.warning(f"{msg}")
            close_open_files()
            h5_path.unlink()
    mth5_obj = MTH5(file_version=file_version)
    mth5_obj.open_mth5(str(h5_path), mode=mode)

    return mth5_obj


def read_back_data(
    mth5_path: str | pathlib.Path,
    station_id: str,
    run_id: str,
    survey: str | None = None,
    close_mth5: bool = True,
    return_objects: list[str] | None = None,
) -> dict[str, Any]:
    """
    Read station/run data from MTH5 file for testing and validation.

    Helper function to confirm MTH5 file accessibility and validate
    that data dimensions match expectations.

    Parameters
    ----------
    mth5_path : str | pathlib.Path
        Full path to MTH5 file to read.
    station_id : str
        Station identifier (e.g., 'PKD', 'MT001').
    run_id : str
        Run identifier (e.g., '001', '1').
    survey : str, optional
        Survey identifier. Required for file version 0.2.0.
    close_mth5 : bool, default True
        Whether to close MTH5 object after reading.
        Set to False if you need to access the object later.
    return_objects : list of str, optional
        Specifies what objects to return. Options:
        - 'run': RunGroup object
        - 'run_ts': RunTS time series object
        If None, returns empty dict with only mth5_obj if close_mth5=False.

    Returns
    -------
    dict
        Dictionary containing requested objects:
        - 'run': RunGroup (if 'run' in return_objects)
        - 'run_ts': RunTS (if 'run_ts' in return_objects)
        - 'mth5_obj': MTH5 (if close_mth5=False)

    Warnings
    --------
    If close_mth5=False, the MTH5 object must be manually closed
    to avoid resource leaks.

    Notes
    -----
    This is primarily a testing utility. Data shape is logged to console.

    Examples
    --------
    Read run data and close immediately::

        >>> result = read_back_data(
        ...     '/path/to/file.mth5',
        ...     'PKD',
        ...     '001',
        ...     return_objects=['run_ts']
        ... )
        >>> ts = result['run_ts']
        >>> print(ts.dataset.shape)

    Read data and keep MTH5 object open::

        >>> result = read_back_data(
        ...     '/path/to/file.mth5',
        ...     'MT001',
        ...     '1',
        ...     survey='survey_01',
        ...     close_mth5=False,
        ...     return_objects=['run', 'run_ts']
        ... )
        >>> run = result['run']
        >>> m = result['mth5_obj']
        >>> # ... use objects ...
        >>> m.close_mth5()

    TODO: add path_or_mth5_decorator to this function
    """
    if return_objects is None:
        return_objects = []
    processing_config: dict[str, Any] = {}
    processing_config["mth5_path"] = str(mth5_path)
    processing_config["local_station_id"] = station_id
    config = processing_config
    m = initialize_mth5(config["mth5_path"], mode="r")
    local_run_obj = m.get_run(config["local_station_id"], run_id, survey=survey)
    local_run_ts = local_run_obj.to_runts()
    data_array = local_run_ts.dataset.to_array()
    logger.info(f"data shape = {data_array.shape}")

    return_dict: dict[str, Any] = {}
    if "run" in return_objects:
        return_dict["run"] = local_run_obj
    if "run_ts" in return_objects:
        return_dict["run_ts"] = local_run_ts
    if close_mth5:
        m.close_mth5()
    else:
        return_dict["mth5_obj"] = m
    return return_dict


def get_compare_dict(input_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Remove MTH5-specific metadata attributes for comparison.

    Removes internal attributes added by MTH5 that may interfere
    with dictionary comparisons between metadata objects.

    Parameters
    ----------
    input_dict : dict
        Dictionary to clean, typically metadata dictionary.

    Returns
    -------
    dict
        Dictionary with MTH5 internal attributes removed.
        Original dict is modified in-place.

    Notes
    -----
    Removed attributes:
    - hdf5_reference: HDF5 object reference (internal)
    - mth5_type: MTH5 data type marker (internal)

    Examples
    --------
    Clean metadata dictionary before comparison::

        >>> metadata = {
        ...     'id': 'station_001',
        ...     'latitude': 45.5,
        ...     'hdf5_reference': <h5py reference>,
        ...     'mth5_type': 'Station'
        ... }
        >>> clean = get_compare_dict(metadata)
        >>> print(clean)
        {'id': 'station_001', 'latitude': 45.5}

    Safe to call with incomplete dicts::

        >>> metadata = {'id': 'station_001'}
        >>> clean = get_compare_dict(metadata)  # No error if keys absent
    """
    for key in ["hdf5_reference", "mth5_type"]:
        try:
            input_dict.pop(key)
        except KeyError:
            pass

    return input_dict


@path_or_mth5_object
def station_in_mth5(
    m: str | pathlib.Path | MTH5,
    station_id: str,
    survey_id: str | None = None,
) -> bool:
    """
    Check if a station exists in MTH5 file.

    Determines whether a station with the given ID is present
    in the MTH5 file using the groups list.

    Parameters
    ----------
    m : str | pathlib.Path | MTH5
        Path to MTH5 file or MTH5 object.
    station_id : str
        Station identifier (e.g., 'PKD', 'MT001').
    survey_id : str, optional
        Survey identifier. Required for file version 0.2.0,
        ignored for version 0.1.0.

    Returns
    -------
    bool
        True if station exists, False otherwise.

    Raises
    ------
    NotImplementedError
        If file version is not 0.1.0 or 0.2.0.

    Notes
    -----
    File version 0.1.0 has global stations group.
    File version 0.2.0 has per-survey stations groups.

    Alternative method: Use channel_summary DataFrame::

        df = m.channel_summary.to_dataframe()
        station_exists = station_id in df['Station'].unique()

    Examples
    --------
    Check if station exists (file version 0.1.0)::

        >>> exists = station_in_mth5('/path/to/file.mth5', 'PKD')
        >>> print(exists)
        True

    Check in version 0.2.0 with survey ID::

        >>> exists = station_in_mth5(
        ...     '/path/to/file.mth5',
        ...     'MT001',
        ...     survey_id='survey_01'
        ... )
    """
    file_version = m.file_version  # type: ignore # decorated by path_or_mth5_object
    if file_version == "0.1.0":
        station_exists = station_id in m.stations_group.groups_list  # type: ignore # decorated by path_or_mth5_object
    elif file_version == "0.2.0":
        survey = m.get_survey(survey_id)  # type: ignore # decorated by path_or_mth5_object
        station_exists = station_id in survey.stations_group.groups_list
    else:
        msg = f"MTH5 file_version {file_version} not understood"
        logger.error(msg)
        raise NotImplementedError(msg)
    return station_exists


@path_or_mth5_object
def survey_in_mth5(m: str | pathlib.Path | MTH5, survey_id: str | None = None) -> bool:
    """
    Check if a survey exists in MTH5 file.

    Determines whether a survey with the given ID exists in the MTH5 file.
    Behavior varies by file version: 0.1.0 has a single survey, while
    0.2.0 supports multiple surveys.

    Parameters
    ----------
    m : str | pathlib.Path | MTH5
        Path to MTH5 file or MTH5 object.
    survey_id : str, optional
        Survey identifier. For file version 0.1.0, compared against the
        global survey ID. For version 0.2.0, checked in surveys group.

    Returns
    -------
    bool
        True if survey exists, False otherwise.

    Raises
    ------
    NotImplementedError
        If file version is not 0.1.0 or 0.2.0.

    Notes
    -----
    File version 0.1.0 has a single survey with fixed ID.
    File version 0.2.0 supports multiple named surveys.

    Alternative method: Use channel_summary DataFrame::

        df = m.channel_summary.to_dataframe()
        surveys = df['Survey'].unique()
        survey_exists = survey_id in surveys

    Examples
    --------
    Check if survey exists (file version 0.1.0)::

        >>> exists = survey_in_mth5('/path/to/file.mth5', 'survey_01')
        >>> print(exists)
        True

    Check in version 0.2.0::

        >>> exists = survey_in_mth5('/path/to/file.mth5', survey_id='MT')
        >>> if exists:
        ...     print(f"Survey MT found in file")
    """
    file_version = m.file_version  # type: ignore # decorated by path_or_mth5_object
    if file_version == "0.1.0":
        survey_metadata = m.survey_group.metadata  # type: ignore
        survey_exists = survey_metadata.id == survey_id  # type: ignore
    elif file_version == "0.2.0":
        survey_exists = survey_id in m.surveys_group.groups_list  # type: ignore
    else:
        msg = f"MTH5 file_version {file_version} not understood"
        logger.error(msg)
        raise NotImplementedError(msg)
    return survey_exists
