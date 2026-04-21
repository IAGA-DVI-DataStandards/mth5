# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:58:44 2022

@author: jpeacock
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Imports
# =============================================================================
import requests
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Magnetic, Run, Station, Survey
from mt_timeseries import ChannelTS, RunTS

from mth5 import __version__ as mth5_version
from mth5.mth5 import MTH5


# =============================================================================

"https://geomag.usgs.gov/ws/data/?id=FRN&type=adjusted&elements=H&sampling_period=1&format=json&starttime=2020-06-02T19:00:00Z&endtime=2020-06-02T22:07:46Z"


class GeomagClient:
    """
    Get geomagnetic data from observatories.

    key words

    - **observatory**: Geogmangetic observatory ID
    - **type**: type of data to get 'adjusted'
    - **start**: start date time to request UTC
    - **end**: end date time to request UTC
    - **elements**: components to get
    - **sampling_period**: samples between measurements in seconds
    - **format**: JSON or IAGA2002

    .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data

    """

    def __init__(self, **kwargs):
        self._base_url = r"https://geomag.usgs.gov/ws/data/"
        self._timeout = 120

        self._valid_observatories = [
            "BDT",
            "BOU",
            "TST",
            "BRW",
            "BRT",
            "BSL",
            "CMO",
            "CMT",
            "DED",
            "DHT",
            "FRD",
            "FRN",
            "GUA",
            "HON",
            "NEW",
            "SHU",
            "SIT",
            "SJG",
            "TUC",
            "USGS",
            "BLC",
            "BRD",
            "CBB",
            "EUA",
            "FCC",
            "IQA",
            "MEA",
            "OTT",
            "RES",
            "SNK",
            "STJ",
            "VIC",
            "YKC",
            "HAD",
            "HER",
            "KAK",
        ]

        self._valid_elements = [
            "D",
            "DIST",
            "DST",
            "E",
            "E-E",
            "E-N",
            "F",
            "G",
            "H",
            "SQ",
            "SV",
            "UK1",
            "UK2",
            "UK3",
            "UK4",
            "X",
            "Y",
            "Z",
        ]

        self._ch_map = {"x": "hx", "y": "hy", "z": "hz"}

        self._valid_sampling_periods = [1, 60, 3600]
        self._valid_output_formats = ["json", "iaga2002"]

        self.type = "adjusted"
        self.sampling_period = 1
        self.elements = ["x", "y"]
        self.format = "json"
        self._timeout = 120
        self.observatory = "FRN"
        self._max_length = 144000
        self.start = None
        self.end = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def user_agent(self):
        """
        User agent for the URL request

        :return: DESCRIPTION
        :rtype: TYPE

        """
        encoding = sys.getdefaultencoding() or "UTF-8"
        platform_ = platform.platform().encode(encoding).decode("ascii", "ignore")

        return f"MTH5 v{mth5_version} ({platform_}, Python {platform.python_version()})"

    @property
    def observatory(self):
        return self._id

    @observatory.setter
    def observatory(self, value):
        """
        make sure value is in accepted list of observatories

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if not isinstance(value, str):
            raise TypeError("input must be a string")

        value = value.upper()
        if value not in self._valid_observatories:
            raise ValueError(
                f"{value} not in accepted observatories see "
                "https://www.usgs.gov/tools/web-service-geomagnetism-data "
                "for more information."
            )

        self._id = value

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value):
        """
        make sure elements are in accepted elements
        """

        if isinstance(value, str):
            if value.count(",") > 0:
                value = [item.strip() for item in value.split(",")]

        if not isinstance(value, list):
            value = [value]

        elements = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"{item} in element list must be a string")
            item = item.upper()
            if item not in self._valid_elements:
                raise ValueError(
                    f"{item} is not an accepted element see "
                    "https://www.usgs.gov/tools/web-service-geomagnetism-data "
                    "for more information."
                )
            elements.append(item)

        self._elements = elements

    @property
    def sampling_period(self):
        return self._sampling_period

    @sampling_period.setter
    def sampling_period(self, value):
        """
        validate sample period value

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"{value} must be able to convert to an integer.")

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{value} must be an integer or float not type({type(value)}"
            )

        if value not in self._valid_sampling_periods:
            raise ValueError(f"{value} must be in [1, 60, 3600]")

        self._sampling_period = value

    @property
    def start(self):
        return f"{self._start.iso_no_tz}Z"

    @start.setter
    def start(self, value):
        if value is None:
            self._start = None
        else:
            self._start = MTime(time_stamp=value)

    @property
    def end(self):
        return f"{self._end.iso_no_tz}Z"

    @end.setter
    def end(self, value):
        if value is None:
            self._end = None
        else:
            self._end = MTime(time_stamp=value)

    def get_chunks(self):
        """
        Get the number of chunks of allowable sized to request, includes the elements

        So the max length is the maximum time period that can be requested but includes
        the number of elements in the request.  So if the max length is 172800 seconds
        and the sampling period is 1 second, then the maximum number of elements that can
        be requested is 172800 / (1 * len(elements)).

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.start is not None and self.end is not None:
            dt = np.arange(
                np.datetime64(self._start.iso_no_tz),
                np.datetime64(self._end.iso_no_tz),
                np.timedelta64(
                    int(
                        (round(self._max_length / len(self.elements)))
                        * self.sampling_period
                    ),
                    "s",
                ),
            )
            dt = np.append(dt, np.array([np.datetime64(self._end.iso_no_tz)]))

            dt_request = [
                (
                    f"{MTime(time_stamp=dt[ii]).iso_no_tz}Z",
                    f"{MTime(time_stamp=dt[ii + 1]).iso_no_tz}Z",
                )
                for ii in range(len(dt) - 1)
            ]

            return dt_request

    def _get_request_params(self, start, end):
        """
        Get request parameters

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return {
            "id": self.observatory,
            "type": self.type,
            "elements": ",".join(self.elements),
            "sampling_period": self.sampling_period,
            "format": "json",
            "starttime": start,
            "endtime": end,
        }

    def _get_request_dictionary(self, start, end):
        """
        get the request dictionary

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return {
            "url": self._base_url,
            "headers": {"User-Agent": self.user_agent},
            "params": self._get_request_params(start, end),
            "timeout": self._timeout,
        }

    def _request_data(self, request_dictionary):
        """
        request data from geomag for start and end times using
        `request.get(**request_dictionary)

        :param request_dictionary: DESCRIPTION
        :type request_dictionary: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return requests.get(**request_dictionary)

    def _to_station_metadata(self, request_metadata):
        """

        :param request_metadata: DESCRIPTION
        :type request_metadata: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        sm = Station()
        sm.id = request_metadata["intermagnet"]["imo"]["name"]
        sm.fdsn.id = request_metadata["intermagnet"]["imo"]["iaga_code"]

        coords = request_metadata["intermagnet"]["imo"]["coordinates"]

        if coords[0] > 180:
            sm.location.longitude = coords[0] - 360
        else:
            sm.location.longitude = coords[0]
        sm.location.latitude = coords[1]
        sm.location.elevation = coords[2]
        sm.provenance.creation_time = request_metadata["generated"]

        return sm

    def get_data(self, run_id="001"):
        """
        Get data from geomag client at USGS based on the request.  This might
        have to be done in chunks depending on the request size.  The returned
        output is a json object, which we should turn into a ChannelTS object

        For now read into a pandas dataframe and then into a ChannelTS

        In the future, if the request is large, think about writing
        directly to an MTH5 for better efficiency.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch = dict([(c.lower(), []) for c in self.elements])

        for interval in self.get_chunks():
            request_obj = self._request_data(
                self._get_request_dictionary(interval[0], interval[1])
            )
            if request_obj.status_code == 200:
                request_json = json.loads(request_obj.content)
                for element in request_json["values"]:
                    ch[element["metadata"]["element"].lower()].append(
                        pd.DataFrame(
                            {
                                "data": element["values"],
                            },
                            index=request_json["times"],
                        )
                    )
            else:
                raise IOError(
                    "Could not connect to server. Error code: "
                    f"{request_obj.status_code}"
                )

        survey_metadata = Survey(id="USGS-GEOMAG")
        station_metadata = self._to_station_metadata(request_json["metadata"])
        run_metadata = Run(id=run_id)

        ch_list = []
        for key, df_list in ch.items():
            df = pd.concat(df_list).astype(float)
            ch_metadata = Magnetic()
            ch_metadata.component = self._ch_map[key]
            ch_metadata.sample_rate = 1.0 / self.sampling_period
            ch_metadata.units = "nanotesla"
            if "y" in ch_metadata.component:
                ch_metadata.measurement_azimuth = 90
            ch_metadata.location.latitude = station_metadata.location.latitude
            ch_metadata.location.longitude = station_metadata.location.longitude
            ch_metadata.location.elevation = station_metadata.location.elevation
            ch_metadata.time_period.start = df.index[0]
            ch_metadata.time_period.end = df.index[-1]
            run_metadata.time_period.start = df.index[0]
            run_metadata.time_period.end = df.index[-1]
            station_metadata.time_period.start = df.index[0]
            station_metadata.time_period.end = df.index[-1]
            survey_metadata.time_period.start = df.index[0]
            survey_metadata.time_period.end = df.index[-1]
            ch_list.append(
                ChannelTS(
                    channel_type="magnetic",
                    data=df,
                    channel_metadata=ch_metadata,
                    run_metadata=run_metadata,
                    station_metadata=station_metadata,
                    survey_metadata=survey_metadata,
                )
            )

        return RunTS(
            ch_list,
            run_metadata=run_metadata,
            station_metadata=station_metadata,
            survey_metadata=survey_metadata,
        )


class USGSGeomag:
    def __init__(self, **kwargs):
        self.save_path = Path().cwd()
        self.mth5_filename = None
        self.interact = False
        self.request_columns = [
            "observatory",
            "type",
            "elements",
            "sampling_period",
            "start",
            "end",
        ]

        # parameters of hdf5 file
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1
        self.mth5_file_mode = "w"
        self.mth5_version = "0.2.0"
        self._ch_map = {"x": "hx", "y": "hy", "z": "hz"}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def h5_kwargs(self):
        h5_params = dict(
            file_version=self.mth5_version,
            compression=self.h5_compression,
            compression_opts=self.h5_compression_opts,
            shuffle=self.h5_shuffle,
            fletcher32=self.h5_fletcher32,
            data_level=self.h5_data_level,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key[3:]] = value

        return h5_params

    def validate_request_df(self, request_df):
        """
        Make sure the input request dataframe has the appropriate columns

        :param request_df: request dataframe
        :type request_df: :class:`pandas.DataFrame`
        :return: valid request dataframe
        :rtype: :class:`pandas.DataFrame`

        """

        if not isinstance(request_df, pd.DataFrame):
            if isinstance(request_df, (str, Path)):
                fn = Path(request_df)
                if not fn.exists():
                    raise IOError(f"File {fn} does not exist. Check path")
                request_df = pd.read_csv(fn, infer_datetime_format=True)
            else:
                raise TypeError(
                    f"Request input must be a pandas.DataFrame, not {type(request_df)}."
                )

        if "run" in request_df.columns:
            if sorted(request_df.columns.tolist()) != sorted(
                self.request_columns + ["run"]
            ):
                raise ValueError(
                    f"Request must have columns {', '.join(self.request_columns)}"
                )
        else:
            if sorted(request_df.columns.tolist()) != sorted(self.request_columns):
                raise ValueError(
                    f"Request must have columns {', '.join(self.request_columns)}"
                )

        request_df = self.add_run_id(request_df)

        return request_df

    def add_run_id(self, request_df):
        """
        Add run id to request df

        :param request_df: request dataframe
        :type request_df: :class:`pandas.DataFrame`
        :return: add a run number to unique time windows for each observatory
         at each unique sampling period.
        :rtype: :class:`pandas.DataFrame`

        """

        request_df.start = pd.to_datetime(request_df.start)
        request_df.end = pd.to_datetime(request_df.end)
        request_df["run"] = ""

        for obs in request_df.observatory.unique():
            for sr in request_df.loc[
                request_df.observatory == obs, "sampling_period"
            ].unique():
                sr_df = request_df.loc[
                    (request_df.observatory == obs) & (request_df.sampling_period == sr)
                ].sort_values("start")
                request_df.loc[
                    (request_df.observatory == obs)
                    & (request_df.sampling_period == sr),
                    "run",
                ] = [f"sp{sr}_{ii+1:03}" for ii in range(len(sr_df))]

        return request_df

    def _make_filename(self, save_path, request_df):
        """

        Create filename from the information in the dataframe

        The filename will look like f"usgs_geomag_{obs}_{elements}.h5"

        :param request_df: request dataframe
        :type request_df: :class:`pandas.DataFrame`
        :return: file name derived from dataframe
        :rtype: :class:`pathlib.Path`

        """

        elements = "".join(request_df.elements.explode().unique().tolist())
        obs = "_".join(sorted(request_df.observatory.unique().tolist()))

        save_path = Path(save_path)
        if save_path.is_dir():
            fn = f"usgs_geomag_{obs}_{elements}.h5"
            save_path = save_path.joinpath(fn)

        return save_path

    def make_mth5_from_geomag(self, request_df):
        """
        Download geomagnetic observatory data from USGS webservices into an
        MTH5 using a request dataframe or csv file.

        :param request_df: DataFrame with columns

            - 'observatory'     --> Observatory code
            - 'type'            --> data type [ 'variation' | 'adjusted' | 'quasi-definitive' | 'definitive' ]
            - 'elements'        --> Elements to get [D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4, X, Y, Z]
            - 'sampling_period' --> sample period [ 1 | 60 | 3600 ]
            - 'start'           --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'             --> End time YYYY-MM-DDThh:mm:ss

        :type request_df: :class:`pandas.DataFrame`, str or Path if csv file


        :return: if interact is True an MTH5 object is returned otherwise the
         path to the file is returned
        :rtype: Path or :class:`mth5.mth5.MTH5`

        .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data

        """

        request_df = self.validate_request_df(request_df)

        fn = self._make_filename(self.save_path, request_df)

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(fn, self.mth5_file_mode)

            if self.mth5_version in ["0.1.0"]:
                survey_group = m.survey_group
                survey_group.metadata.id = "USGS-GEOMAG"
            elif self.mth5_version in ["0.2.0"]:
                survey_group = m.add_survey("USGS-GEOMAG")
            else:
                raise ValueError(
                    f"MTH5 version must be [ '0.1.0' | '0.2.0' ] not {self.mth5_version}"
                )

            for row in request_df.itertuples():
                geomag_client = GeomagClient(
                    observatory=row.observatory,
                    type=row.type,
                    elements=row.elements,
                    start=row.start,
                    end=row.end,
                    sampling_period=row.sampling_period,
                    **{"_ch_map": {"x": "hx", "y": "hy", "z": "hz"}},
                )

                run = geomag_client.get_data(run_id=row.run)
                station_group = survey_group.stations_group.add_station(
                    run.station_metadata.id,
                    station_metadata=run.station_metadata,
                )
                run_group = station_group.add_run(
                    run.run_metadata.id, run_metadata=run.run_metadata
                )
                run_group.from_runts(run)
                station_group.update_metadata()
            survey_group.update_metadata()

        if self.interact:
            m.open_mth5(m.filename, self.mth5_file_mode)
            return m
        else:
            return m.filename
