# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock

This module is concerned with creating mth5 files from the synthetic test data
 that originally came from EMTF -- test1.asc and test2.asc.  Each ascii file
 represents five channels of data sampled at 1Hz at a synthetic station.

TODO: Separate the handling of legacy EMTF data files, such as
 reading into a dataframe from oddly delimited data, as well as flipping polarities of
 the electric channels (possibly due to a baked in sign convention error in the legacy
 data), so that a simple dataframe can be passed.  That will make the methods here more
  easily generalize to work with other dataframes.  That would be useful in future when
  we creating synthetic data at arbitrary sample rate.

Development Notes:
 Mirroring the original ascii files are:
 data/test1.h5
 data/test2.h5
 data/test12rr.h5

 Also created are some files with the same data but other channel_nomenclature schemes:
 data/test12rr_LEMI34.h5
 data/test1_LEMI12.h5

 - 20231103: Added an 8Hz up-sampled version of test1.  No spectral content was added
 so the band between the old and new Nyquist frequencies is bogus.



"""
# =============================================================================
# Imports
# =============================================================================

import pathlib
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from mt_metadata.common.comment import Comment
from mt_metadata.processing.aurora import ChannelNomenclature
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic, Survey
from mt_timeseries import ChannelTS, RunTS

from mth5.data.paths import SyntheticTestPaths
from mth5.data.station_config import (
    make_filters,
    make_station_01,
    make_station_02,
    make_station_03,
    make_station_04,
    SyntheticRun,
    SyntheticStation,
)
from mth5.mth5 import MTH5
from mth5.utils.helpers import add_filters


# =============================================================================
np.random.seed(0)

synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path


def create_run_ts_from_synthetic_run(
    run: SyntheticRun, df: pd.DataFrame, channel_nomenclature: ChannelNomenclature
) -> RunTS:
    """
    Loop over channels of synthetic data in df and make ChannelTS objects.

    :param run: One-off data structure with information mth5 needs to initialize. Specifically sample_rate, filters.
    :type run: mth5.data.station_config.SyntheticRun
    :param df: time series data in columns labelled from ["ex", "ey", "hx", "hy", "hz"]
    :type df: pandas.DataFrame
    :param channel_nomenclature : Keyword corresponding to channel nomenclature mapping
    in CHANNEL_MAPS variable from channel_nomenclature.py module in mt_metadata.
    Supported values include ['default', 'lemi12', 'lemi34', 'phoenix123']
    :type channel_nomenclature : string

    :return runts: MTH5 run time series object, data and metadata bound into one.
    :rtype runts: RunTS

    """

    ch_list = []
    for i_col, col in enumerate(df.columns):
        data = df[col].values
        if col in channel_nomenclature.ex_ey:
            channel_metadata = Electric()
            channel_metadata.units = "milliVolt per kilometer"
        elif col in channel_nomenclature.hx_hy_hz:
            channel_metadata = Magnetic()
            channel_metadata.units = "nanotesla"
        else:
            msg = f"column {col} not in channel_nomenclature {channel_nomenclature}"
            logger.error(msg)
            raise ValueError(msg)

        channel_metadata.component = col
        channel_metadata.channel_number = i_col  # not required
        channel_metadata.sample_rate = run.run_metadata.sample_rate
        channel_metadata.time_period.start = run.run_metadata.time_period.start
        chts = ChannelTS(
            channel_type=channel_metadata.type,  # "electric" or "magnetic"
            data=data,
            channel_metadata=channel_metadata.to_dict(),
        )

        # Set dipole properties
        # (Not sure how to pass this in channel_metadata when intializing)
        if col in channel_nomenclature.ex_ey:
            chts.channel_metadata.dipole_length = 50
            if col == channel_nomenclature.ey:
                chts.channel_metadata.measurement_azimuth = 90.0

        elif col in channel_nomenclature.hx_hy_hz:
            if col == channel_nomenclature.hy:
                chts.channel_metadata.measurement_azimuth = 90.0

        # Set filters
        for stage_num, filter_name in enumerate(run.filters[col], start=1):
            applied_filter = AppliedFilter(
                name=filter_name,
                applied=True,
                stage=stage_num,
                comments=Comment(author="system", time_stamp="2024-01-01"),
            )
            chts.channel_metadata.add_filter(applied_filter=applied_filter)

        ch_list.append(chts)

    # make a RunTS object
    if run.run_metadata.sample_rate == 0:
        msg = "Run sample rate cannot be zero, something is fishy, setting to 1.0 Hz"
        logger.warning(msg)
        run.run_metadata.sample_rate = 1.0
    runts = RunTS(array_list=ch_list, run_metadata=run.run_metadata)

    return runts


def get_time_series_dataframe(
    run: SyntheticRun,
    source_folder: Union[pathlib.Path, str],
    add_nan_values: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Returns time series data in a dataframe with columns named for EM field component.

    Up-samples data to run.sample_rate, which is treated as in integer.
    Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
    then no up-sampling takes place.

    TODO: Move noise, and nan addition out of this method.

    :type run: mth5.data.station_config.SyntheticRun
    :param run: Information needed to define/create the run
    :type source_folder: Optional[Union[pathlib.Path, str]]
    :param source_folder: Where to load the ascii time series from.  This overwrites any
    previous value that may have been stored in the SyntheticRun
    :type add_nan_values: bool
    :param add_nan_values: If True, add some NaN, if False, do not add Nan.
    :rtype df: pandas.DataFrame
    :return df: The time series data for the synthetic run

    """
    # point to the ascii time series
    if source_folder:
        run.raw_data_path = source_folder.joinpath(run.raw_data_path.name)

    df = run._get_timeseries_dataframe()

    # add noise if requested
    for col in run.channels:
        if run.noise_scalars[col]:
            df[col] += run.noise_scalars[col] * np.random.randn(len(df))

    # add nan if requested
    if add_nan_values:
        for col in run.channels:
            for [ndx, num_nan] in run.nan_indices[col]:
                df.loc[ndx : ndx + num_nan, col] = np.nan
    return df


def create_mth5_synthetic_file(
    station_cfgs: List[SyntheticStation],
    mth5_name: Union[pathlib.Path, str],
    target_folder: Optional[Union[pathlib.Path, str]] = "",
    source_folder: Union[pathlib.Path, str] = "",
    plot: bool = False,
    add_nan_values: bool = False,
    file_version: Literal["0.1.0", "0.2.0"] = "0.1.0",
    force_make_mth5: bool = True,
    survey_metadata: Optional[Survey] = None,
):
    """
    Creates an MTH5 from synthetic data.

    Development Notes:
     20250203: This function could be made more general, so that it operates on dataframes and legacy emtf ascii files.

    :param station_cfgs: Iterable of objects of type SyntheticStation. These are one-off
    data structure used to hold information mth5 needs to initialize, specifically
    sample_rate, filters, etc.
    :type station_cfgs: List[SyntheticStation]
    :param mth5_name: Where the mth5 will be stored.  This is generated by the station_config,
    but may change in this method based on add_nan_values or channel_nomenclature
    :type mth5_name: Union[pathlib.Path, str]
    :param target_folder: Where the mth5 file will be stored
    :type target_folder: Optional[Union[pathlib.Path, str]]
    :param source_folder:  Where the ascii source data are stored
    :type source_folder: Optional[Union[pathlib.Path, str]] = "",
    :param plot: Set to false unless you want to look at a plot of the time series
    :type plot: bool
    :param add_nan_values: If true, some np.nan are sprinkled into the time series.  Intended to be used for tests.
    :type add_nan_values: bool
    :param file_version: One of the supported mth5 file versions.  This is the version of mth5 to create.
    :type file_version: Literal["0.1.0", "0.2.0"] = "0.1.0",
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :type force_make_mth5: bool
    :param survey_metadata: Option to provide survey metadata, otherwise it will be created.
    :type survey_metadata: Survey
    :return: The path to the stored h5 file.
    :rtype: mth5_path: pathlib.Path

    """
    nomenclatures = [x.channel_nomenclature.keyword for x in station_cfgs]
    unconventional_nomenclatures = [x for x in nomenclatures if x.lower() != "default"]
    if unconventional_nomenclatures:
        nomenclature_str = "_".join(unconventional_nomenclatures)
    else:
        nomenclature_str = ""

    # determine the path to the file that will be created
    target_folder = _get_target_folder(target_folder=target_folder)
    mth5_path = target_folder.joinpath(mth5_name)
    mth5_path = _update_mth5_path(
        mth5_path, add_nan_values, channel_nomenclature=nomenclature_str
    )

    # Only create file if needed
    if not force_make_mth5:
        if mth5_path.exists():
            return mth5_path

    # create survey metadata:
    if not survey_metadata:
        survey_id = "EMTF Synthetic"
        survey_metadata = Survey()
        survey_metadata.id = survey_id

    # open output h5
    with MTH5(file_version=file_version) as m:
        m.open_mth5(mth5_path, mode="w")
        _add_survey(m, survey_metadata)

        for station_cfg in station_cfgs:
            station_group = m.add_station(
                station_cfg.station_metadata.id,
                station_metadata=station_cfg.station_metadata,
                survey=survey_id,
            )

            for run in station_cfg.runs:
                # run is object of type SyntheticRun
                df = get_time_series_dataframe(
                    run=run, source_folder=source_folder, add_nan_values=add_nan_values
                )

                # TODO: Add handling for noise, nan, and upsampling here
                #  (They don't belong in get_time_Series_dataframe()

                # cast to run_ts
                # TODO: This could be
                #  synthetic_run.to_run_ts(df)
                # but channel types for each column name must come from the Station level.
                runts = create_run_ts_from_synthetic_run(
                    run, df, channel_nomenclature=station_cfg.channel_nomenclature
                )
                runts.station_metadata.id = station_group.metadata.id

                # plot the data
                if plot:
                    runts.plot()

                run_group = station_group.add_run(run.run_metadata.id)
                run_group.from_runts(runts)

        # add filters
        active_filters = make_filters(as_list=True)
        add_filters(m, active_filters, survey_id)

    logger.info(f"Created synthetic mth5 file: {mth5_path}")

    return mth5_path


def _get_target_folder(target_folder: Optional[Union[pathlib.Path, str]] = ""):
    """
    Return the target folder where an mth5 file will be created

    :param target_folder: This is where the mth5 will be created.  If this argument is null,
    then set to MTH5_PATH
    :type target_folder: Optional[Union[pathlib.Path, str]]

    :return: the path where an mth5 file will be created
    :rtype: pathlib.Path

    """
    # Handle path and file name conventions
    if not target_folder:
        msg = "No target folder provided for making mth5 file"
        logger.warning(msg)
        msg = f"Setting target folder to {MTH5_PATH}"
        logger.info(msg)
        target_folder = MTH5_PATH

    if isinstance(target_folder, str):
        target_folder = pathlib.Path(target_folder)

    try:
        target_folder.mkdir(exist_ok=True, parents=True)
    except OSError:
        msg = "MTH5 maybe installed on a read-only file system"
        msg = f"{msg}: try setting `target_folder` argument when calling create_mth5_synthetic_file"
        logger.error(msg)
    return target_folder


def create_test1_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1".

    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type file_version: str
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[Union[str, pathlib.Path]]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[Union[str, pathlib.Path]]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file

    """
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]

    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    return mth5_path


def create_test2_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test2".

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    mth5_name = station_02_params.mth5_name
    station_params = [
        station_02_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test1_h5_with_nan(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1" with some nan values.

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        add_nan_values=True,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test12rr_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file with data from two stations station named "test1" and "test2".

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    station_params = [station_01_params, station_02_params]
    mth5_name = "test12rr.h5"
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        file_version=file_version,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    mth5_path = pathlib.Path(mth5_path)
    return mth5_path


def create_test3_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test3".
    This example has several runs and can be used to test looping over runs.

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_03_params = make_station_03(channel_nomenclature=channel_nomenclature)
    station_params = [
        station_03_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        station_params[0].mth5_name,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test4_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1", data are up-sampled to 8Hz from
    original 1 Hz.

    Note: Because the 8Hz data are derived from the 1Hz, only frequencies below 0.5Hz
    will have valid TFs that yield the apparent resistivity of the synthetic data (100 Ohm-m).

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_04_params = make_station_04(channel_nomenclature=channel_nomenclature)
    mth5_path = create_mth5_synthetic_file(
        [
            station_04_params,
        ],
        station_04_params.mth5_name,
        plot=False,
        file_version=file_version,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    return mth5_path


def _add_survey(m: MTH5, survey_metadata: Survey) -> None:
    """
    :type m: mth5.mth5.MTH5
    :param m: The mth5 object to get/set survey_id with
    :type survey_metadata: mt_metadata.timeseries.Survey
    :param survey_metadata: The survey metadata in mt_metadata container

    """
    if m.file_version == "0.1.0":
        # "no need to pass survey id in v 0.1.0 -- just the metadata"
        m.survey_group.update_metadata(survey_metadata.to_dict())
    elif m.file_version == "0.2.0":
        m.add_survey(survey_metadata.id, survey_metadata)
    else:
        msg = f"unexpected MTH5 file_version = {m.file_version}"
        raise NotImplementedError(msg)
    return


def _update_mth5_path(
    mth5_path: pathlib.Path, add_nan_values: bool, channel_nomenclature: str
) -> pathlib.Path:
    """

    Modify the name of output h5 file based on wheter or not nan-data are included
     as well as channel_nomenclature if not default for all stations.

    :param mth5_path:
    :param add_nan_values:
    :param channel_nomenclature: designator for the channel nomenclatures in the mth5
    :type channel_nomenclature: str

    :return: TODO
    :rtype: pathlib.Path

    """

    path_str = mth5_path.__str__()
    if add_nan_values:
        path_str = path_str.replace(".h5", "_nan.h5")
    if channel_nomenclature:
        if channel_nomenclature != "default":
            path_str = path_str.replace(".h5", f"_{channel_nomenclature}.h5")
    return pathlib.Path(path_str)


def main(file_version="0.1.0"):
    """Allow the module to be called from the command line"""
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version, channel_nomenclature="lemi12")
    create_test3_h5(file_version=file_version)
    create_test4_h5(file_version=file_version)


if __name__ == "__main__":
    main(file_version="0.2.0")
    main(file_version="0.1.0")
