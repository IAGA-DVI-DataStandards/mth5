# -*- coding: utf-8 -*-
"""
LEMI Client
===========

Client for creating MTH5 files from LEMI instrument data.
Supports both LEMI-424 (long period, .txt) and LEMI-423 (broadband, .B423) files.

Created on Fri Oct 11 10:57:54 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import warnings

from mt_io.lemi import LEMICollection

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class LEMIClient(ClientBase):
    """
    Client for creating MTH5 files from LEMI instrument data.

    Supports both LEMI-424 and LEMI-423 instruments:
    - **LEMI-424**: Long period data (.txt files, 1 Hz)
    - **LEMI-423**: Broadband data (.B423 files, 4000/2000/1000/500/250 Hz)

    :param data_path: Path to directory containing LEMI files
    :type data_path: str or pathlib.Path
    :param save_path: Path to save MTH5 file, defaults to data_path
    :type save_path: str or pathlib.Path, optional
    :param sample_rates: Sample rates to include, defaults to [1]
     Valid rates:
     - LEMI-424: [1] (long period only)
     - LEMI-423: [4000, 2000, 1000, 500, 250] (broadband, auto-detected from tick counter)
     To include all LEMI-423 rates: [4000, 2000, 1000, 500, 250]
    :type sample_rates: list, optional
    :param mth5_filename: Name of output MTH5 file, defaults to "from_lemi.h5"
    :type mth5_filename: str, optional

    :Example:

        >>> from mth5.clients import LEMIClient
        >>> # LEMI-424 data
        >>> lc = LEMIClient("/path/to/lemi424/data", sample_rates=[1])
        >>> mth5_path = lc.make_mth5_from_lemi("Survey01", "MT001")
        >>>
        >>> # LEMI-423 data at 1000 Hz
        >>> lc = LEMIClient("/path/to/lemi423/data", sample_rates=[1000])
        >>> mth5_path = lc.make_mth5_from_lemi("Survey01", "MT002")
        >>>
        >>> # LEMI-423 data, unknown sample rate (include all possible)
        >>> lc = LEMIClient("/path/to/lemi423/data",
        ...                 sample_rates=[4000, 2000, 1000, 500, 250])
        >>> mth5_path = lc.make_mth5_from_lemi("Survey01", "MT003")

    """

    def __init__(
        self,
        data_path,
        save_path=None,
        sample_rates=[1],
        mth5_filename="from_lemi.h5",
        **kwargs,
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.collection = LEMICollection(self.data_path)

    def make_mth5_from_lemi(self, survey_id, station_id, **kwargs):
        """
        Create an MTH5 file from LEMI instrument data.

        Works with both LEMI-424 (.txt) and LEMI-423 (.B423) files.

        :param survey_id: Survey identifier
        :type survey_id: str
        :param station_id: Station identifier
        :type station_id: str
        :param kwargs: Additional keyword arguments passed to collection
        :type kwargs: dict
        :return: Path to created MTH5 file
        :rtype: pathlib.Path

        :Example:

            >>> from mth5.clients import LEMIClient
            >>> lc = LEMIClient("/data/lemi423", sample_rates=[1000])
            >>> mth5_file = lc.make_mth5_from_lemi("MySurvey", "MT001")
            >>> print(f"Created: {mth5_file}")

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        self.collection.survey_id = survey_id
        self.collection.station_id = station_id

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(station_id)
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_file(run_df.fn.to_list())
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path

    # Backward compatibility alias
    def make_mth5_from_lemi424(self, survey_id, station_id, **kwargs):
        """
        Backward compatibility alias for make_mth5_from_lemi().

        .. deprecated:: 0.4.0
            Use :meth:`make_mth5_from_lemi` instead. This method will be
            removed in version 1.0.0.
        """
        warnings.warn(
            "make_mth5_from_lemi424() is deprecated, use make_mth5_from_lemi() instead. "
            "This method will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.make_mth5_from_lemi(survey_id, station_id, **kwargs)


# Backward compatibility alias
LEMI424Client = LEMIClient
