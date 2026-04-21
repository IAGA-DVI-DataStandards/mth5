# -*- coding: utf-8 -*-
"""
Make MTH5
============

This module provides helper functions to make MTH5 file from various clients

Supported Clients include:

    * FDSN (through Obspy)
    * Science Base (TODO)
    * NCI - Australia (TODO)
    * Phoenix MTU-5C
    * Zen
    * LEMI 424
    * Phoenix legacy MTU (TODO)
    * Metronix Geophysics
    * USGS Geomagnetism


Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import pandas as pd

from . import (
    FDSN,
    LEMI424Client,
    MetronixClient,
    NIMSClient,
    PhoenixClient,
    USGSGeomag,
    ZenClient,
)


# =============================================================================


class MakeMTH5:
    """
    Factory class for creating MTH5 files from various data sources.

    This class provides class methods to create MTH5 files from different
    magnetotelluric data acquisition systems and data repositories.

    Parameters
    ----------
    mth5_version : str, default "0.2.0"
        MTH5 file format version
    interact : bool, default False
        If True, keep file open for interactive use. If False, close file after creation.
    save_path : str or Path, optional
        Directory path to save MTH5 file. If None, uses current working directory.
    **kwargs : dict
        Additional keyword arguments for HDF5 file parameters. Any parameter
        starting with 'h5' will be used for HDF5 configuration.

    Attributes
    ----------
    h5_compression : str, default "gzip"
        HDF5 compression algorithm
    h5_compression_opts : int, default 4
        Compression level (0-9 for gzip)
    h5_shuffle : bool, default True
        Enable byte shuffle filter for better compression
    h5_fletcher32 : bool, default True
        Enable Fletcher32 checksum for data integrity
    h5_data_level : int, default 1
        Data processing level indicator

    Examples
    --------
    Create a basic MakeMTH5 instance:

    >>> from mth5.clients import MakeMTH5
    >>> maker = MakeMTH5(save_path="/path/to/save")
    >>> print(maker)
    MakeMTH5 Attibutes:
        mth5_version: 0.2.0
        h5_compression: gzip
        ...

    Create with custom compression:

    >>> maker = MakeMTH5(
    ...     save_path="/data/mt",
    ...     h5_compression="lzf",
    ...     h5_shuffle=False
    ... )

    See Also
    --------
    mth5.mth5.MTH5 : Main MTH5 file interface
    """

    def __init__(
        self,
        mth5_version: str = "0.2.0",
        interact: bool = False,
        save_path: str | Path | None = None,
        **kwargs,
    ):
        self.mth5_version = mth5_version
        self.interact = interact
        self.save_path = save_path
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1
        self.mth5_file_mode = "w"
        self.mth5_filename = "make_mth5.h5"

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.save_path is None:
            self.save_path = Path().cwd()

    def __str__(self):
        lines = ["MakeMTH5 Attibutes:"]
        for key, value in self.get_h5_kwargs().items():
            lines.append(f"\t{key}: {value}")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    @property
    def save_path(self) -> Path:
        """Get the save path as a Path object."""
        return Path(self._save_path)

    @save_path.setter
    def save_path(self, value: str | Path | None):
        """Set the save path, converting to Path if necessary."""
        if value is None:
            self._save_path = Path().cwd()
        else:
            self._save_path = Path(value)
            if "." in self._save_path.name:
                self.mth5_filename = self._save_path.name
                self._save_path = self._save_path.parent
            else:
                if not self._save_path.exists():
                    self._save_path.mkdir(parents=True, exist_ok=True)

    def get_h5_kwargs(self) -> dict:
        """
        Extract HDF5-related keyword arguments from instance attributes.

        Returns
        -------
        dict
            Dictionary of HDF5 configuration parameters including version,
            compression settings, shuffle, fletcher32, and data level.

        Examples
        --------
        >>> maker = MakeMTH5(h5_compression="lzf", h5_data_level=2)
        >>> kwargs = maker.get_h5_kwargs()
        >>> print(kwargs["h5_compression"])
        lzf
        """
        h5_params = dict(
            mth5_version=self.mth5_version,
            h5_compression=self.h5_compression,
            h5_compression_opts=self.h5_compression_opts,
            h5_shuffle=self.h5_shuffle,
            h5_fletcher32=self.h5_fletcher32,
            h5_data_level=self.h5_data_level,
            mth5_file_mode=self.mth5_file_mode,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key] = value

        return h5_params

    @classmethod
    def from_fdsn_client(cls, request_df: pd.DataFrame, client: str = "IRIS", **kwargs):
        """
        Create MTH5 file from FDSN data service.

        Pull data from an FDSN archive like IRIS using ObsPy clients.
        The request DataFrame specifies which data to download.

        Parameters
        ----------
        request_df : pd.DataFrame
            DataFrame with columns:

            - 'network' : str
                FDSN Network code (e.g., 'IU', 'TA')
            - 'station' : str
                FDSN Station code (e.g., 'ANMO', 'CAS04')
            - 'location' : str
                FDSN Location code (e.g., '00', '')
            - 'channel' : str
                FDSN Channel code (e.g., 'LFE', 'BHZ')
            - 'start' : str
                Start time in format 'YYYY-MM-DDThh:mm:ss'
            - 'end' : str
                End time in format 'YYYY-MM-DDThh:mm:ss'

        client : str, default "IRIS"
            FDSN client name (e.g., 'IRIS', 'USGS', 'NCEDC')
        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=4).

        Returns
        -------
        mth5.mth5.MTH5 or Path
            MTH5 object if interact=True, otherwise Path to created file

        Raises
        ------
        AttributeError
            If the input DataFrame is not properly formatted
        ValueError
            If the DataFrame column values are invalid

        Notes
        -----
        If any column value is blank, any matching value will be searched.
        For example, leaving 'station' blank will return all stations within
        the specified time range.

        Examples
        --------
        Create a request DataFrame and download data:

        >>> import pandas as pd
        >>> from mth5.clients import MakeMTH5
        >>>
        >>> # Define request
        >>> request_df = pd.DataFrame({
        ...     'network': ['IU'],
        ...     'station': ['ANMO'],
        ...     'location': ['00'],
        ...     'channel': ['LF*'],
        ...     'start': ['2020-01-01T00:00:00'],
        ...     'end': ['2020-01-02T00:00:00']
        ... })
        >>>
        >>> # Create MTH5 with custom compression
        >>> mth5_obj = MakeMTH5.from_fdsn_client(
        ...     request_df,
        ...     client='IRIS',
        ...     h5_compression_opts=1
        ... )

        See Also
        --------
        from_fdsn_miniseed_and_stationxml : Create from existing files
        obspy.clients.fdsn : ObsPy FDSN client documentation
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()
        kw_dict["mth5_filename"] = maker.mth5_filename

        fdsn_client = FDSN(client=client, **kw_dict)

        mth5_object = fdsn_client.make_mth5_from_fdsn_client(
            request_df, path=maker.save_path, interact=maker.interact
        )

        return mth5_object

    @classmethod
    def from_fdsn_miniseed_and_stationxml(
        cls,
        station_xml_path: str | Path,
        miniseed_files: str | Path | list[str | Path],
        save_path: str | Path | None = None,
        **kwargs,
    ):
        """
        Create MTH5 from existing StationXML and miniSEED files.

        Use this method when you already have StationXML and miniSEED files
        downloaded from an FDSN client or created locally.

        Parameters
        ----------
        station_xml_path : str, Path, or obspy.Inventory
            Full path to StationXML file or an ObsPy Inventory object
        miniseed_files : str, Path, list, or obspy.Stream
            List of miniSEED file paths or ObsPy Stream objects. Can also
            be a single file path or Stream object.
        save_path : str or Path, optional
            Directory to save new MTH5 file. If None, saves to current
            working directory.
        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip').

        Returns
        -------
        Path
            Path to created MTH5 file. Filename format is
            {network}_{station}.h5 based on unique network and station codes.

        Raises
        ------
        TypeError
            If inputs are not of correct type

        Examples
        --------
        Create MTH5 from existing files:

        >>> from mth5.clients import MakeMTH5
        >>> from pathlib import Path
        >>>
        >>> # Define file paths
        >>> station_xml = Path("data/station.xml")
        >>> miniseed = [
        ...     Path("data/IU.ANMO.00.LFE.mseed"),
        ...     Path("data/IU.ANMO.00.LFN.mseed")
        ... ]
        >>>
        >>> # Create MTH5
        >>> mth5_path = MakeMTH5.from_fdsn_miniseed_and_stationxml(
        ...     station_xml,
        ...     miniseed,
        ...     save_path="output",
        ...     h5_compression="lzf"
        ... )
        >>> print(mth5_path)
        output/IU_ANMO.h5

        Using ObsPy objects directly:

        >>> from obspy import read, read_inventory
        >>>
        >>> inventory = read_inventory("station.xml")
        >>> stream = read("data/*.mseed")
        >>>
        >>> mth5_path = MakeMTH5.from_fdsn_miniseed_and_stationxml(
        ...     inventory,
        ...     stream,
        ...     save_path="output"
        ... )

        See Also
        --------
        from_fdsn_client : Download and create in one step
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()
        kw_dict["mth5_filename"] = maker.mth5_filename

        fdsn_client = FDSN(**kw_dict)

        return fdsn_client.make_mth5_from_inventory_and_streams(
            station_xml_path, miniseed_files, save_path=save_path
        )

    @classmethod
    def from_usgs_geomag(cls, request_df: pd.DataFrame | str | Path, **kwargs):
        """
        Create MTH5 from USGS geomagnetic observatory data.

        Downloads geomagnetic observatory data from USGS webservices into an
        MTH5 file using a request DataFrame or CSV file.

        Parameters
        ----------
        request_df : pd.DataFrame, str, or Path
            Request definition as DataFrame or path to CSV file. Required columns:

            * **observatory** : str - Observatory code (e.g., 'BOU', 'FRN')
            * **type** : str - Data type: 'variation', 'adjusted',
              'quasi-definitive', or 'definitive'
            * **elements** : str - Geomagnetic elements to retrieve:
              D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4,
              X, Y, Z
            * **sampling_period** : int - Sample period in seconds: 1, 60, or 3600
            * **start** : str - Start time in YYYY-MM-DDThh:mm:ss format (UTC)
            * **end** : str - End time in YYYY-MM-DDThh:mm:ss format (UTC)

        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=1).

        Returns
        -------
        Path or MTH5
            If interact=False (default), returns Path to created MTH5 file.
            If interact=True, returns MTH5 object with file open.

        Notes
        -----
        See USGS Geomagnetism Data web service for more information:
        https://www.usgs.gov/tools/web-service-geomagnetism-data

        Examples
        --------
        Create MTH5 from USGS Boulder observatory using DataFrame:

        >>> import pandas as pd
        >>> from mth5.clients import MakeMTH5
        >>>
        >>> request = pd.DataFrame([{
        ...     'observatory': 'BOU',
        ...     'type': 'variation',
        ...     'elements': 'XYZF',
        ...     'sampling_period': 1,
        ...     'start': '2020-01-01T00:00:00',
        ...     'end': '2020-01-02T00:00:00'
        ... }])
        >>>
        >>> mth5_path = MakeMTH5.from_usgs_geomag(
        ...     request,
        ...     h5_compression='gzip',
        ...     h5_compression_opts=1
        ... )

        Using CSV file:

        >>> mth5_path = MakeMTH5.from_usgs_geomag('requests.csv')

        Multiple observatories and periods:

        >>> request = pd.DataFrame([
        ...     {'observatory': 'BOU', 'type': 'variation',
        ...      'elements': 'XYZF', 'sampling_period': 1,
        ...      'start': '2020-01-01T00:00:00', 'end': '2020-01-02T00:00:00'},
        ...     {'observatory': 'FRN', 'type': 'variation',
        ...      'elements': 'XYZF', 'sampling_period': 60,
        ...      'start': '2020-01-01T00:00:00', 'end': '2020-01-02T00:00:00'}
        ... ])
        >>> mth5_path = MakeMTH5.from_usgs_geomag(request)

        See Also
        --------
        mt_io.usgs_geomag.USGSGeomag : USGS geomagnetic data client
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        geomag_client = USGSGeomag(
            save_path=maker.save_path,
            interact=maker.interact,
            **kw_dict,
        )

        return geomag_client.make_mth5_from_geomag(request_df)

    @classmethod
    def from_zen(
        cls,
        data_path: str | Path,
        sample_rates: list[int] = [4096, 1024, 256],
        calibration_path: str | Path | None = None,
        survey_id: str | None = None,
        combine: bool = True,
        **kwargs,
    ):
        """
        Create MTH5 from Zonge ZEN data files.

        Processes ZEN data files from a directory structure and creates an
        MTH5 file with organized time series data.

        Parameters
        ----------
        data_path : str or Path
            Directory where ZEN data files are stored
        sample_rates : list of int, default [4096, 1024, 256]
            Sample rates to include in Hz
        calibration_path : str or Path, optional
            Path to calibration file (amtant.cal). If None, looks for
            calibration file in data_path.
        survey_id : str, optional
            Survey ID to apply to all stations found under data_path. If None,
            attempts to extract from directory structure.
        combine : bool, default True
            If True, combine multiple runs into single run sampled at 1s
        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=1).
            Use save_path to specify output directory.

        Returns
        -------
        Path
            Path to created MTH5 file

        Notes
        -----
        ZEN data is typically organized with multiple .Z3D files per station.
        The reader processes these files and organizes them into runs based on
        sampling rate and timing.

        When combine=True, all runs are merged into a single continuous run
        sampled at 1 second intervals, which is useful for long-term datasets.

        Examples
        --------
        Create MTH5 from ZEN data directory:

        >>> from mth5.clients import MakeMTH5
        >>> from pathlib import Path
        >>>
        >>> data_dir = Path("data/zen_survey")
        >>> mth5_path = MakeMTH5.from_zen(
        ...     data_dir,
        ...     sample_rates=[4096, 256],
        ...     survey_id="MT001",
        ...     save_path="output"
        ... )

        With calibration file and HDF5 compression:

        >>> mth5_path = MakeMTH5.from_zen(
        ...     "data/zen_survey",
        ...     calibration_path="data/amtant.cal",
        ...     survey_id="MT001",
        ...     combine=False,
        ...     h5_compression="gzip",
        ...     h5_compression_opts=4
        ... )

        Process all sample rates without combining:

        >>> mth5_path = MakeMTH5.from_zen(
        ...     "data/zen_survey",
        ...     sample_rates=[4096, 1024, 256, 64, 4],
        ...     combine=False
        ... )

        See Also
        --------
        mt_io.zen.ZenCollection : ZEN data reader
        """

        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        zc = ZenClient(
            data_path,
            sample_rates=sample_rates,
            save_path=maker.save_path,
            calibration_path=calibration_path,
            **kw_dict,
        )

        return zc.make_mth5_from_zen(survey_id=survey_id, combine=combine, **kwargs)

    @classmethod
    def from_phoenix(
        cls,
        data_path: str | Path,
        mth5_filename: str | None = None,
        save_path: str | Path | None = None,
        sample_rates: list[int] = [150, 24000],
        receiver_calibration_dict: str | Path | dict | None = None,
        sensor_calibration_dict: str | Path | dict | None = None,
        **kwargs,
    ):
        """
        Create MTH5 from Phoenix MTU-5C data files.

        Builds an MTH5 file from Phoenix MTU-5C data with calibration support.
        Requires receiver and sensor calibration files exported from EMPower
        software.

        Parameters
        ----------
        data_path : str or Path
            Directory where Phoenix data files are stored. Can be single station
            or multiple stations.
        mth5_filename : str, optional
            Filename for the MTH5 file. If None, defaults to 'from_phoenix.h5'
        save_path : str or Path, optional
            Directory to save MTH5 file. If None, saves to data_path.
        sample_rates : list of int, default [150, 24000]
            Sample rates to include in Hz
        receiver_calibration_dict : str, Path, or dict, optional
            Receiver calibration specification:

            * str/Path: Directory containing rxcal.json files
            * dict: Keys are receiver IDs, values are paths to rxcal.json files

        sensor_calibration_dict : str, Path, or dict, optional
            Sensor calibration specification:

            * str/Path: Directory containing scal.json files
            * dict: Keys are sensor IDs, values are PhoenixCalibration objects
              or paths to scal.json files

        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip').

        Returns
        -------
        Path
            Path to created MTH5 file

        Notes
        -----
        Phoenix data requires calibration files exported from EMPower software:

        1. Export rxcal files (receiver calibration) to JSON
        2. Export scal files (sensor calibration) to JSON
        3. Place files in accessible directory
        4. Provide directory path or dict mapping to from_phoenix()

        The method automatically matches calibration files with data based on
        receiver and sensor IDs.

        Examples
        --------
        Basic usage with calibration directories:

        >>> from mth5.clients import MakeMTH5
        >>> from pathlib import Path
        >>>
        >>> data_dir = Path("data/phoenix_survey")
        >>> cal_dir = Path("calibrations")
        >>>
        >>> mth5_path = MakeMTH5.from_phoenix(
        ...     data_dir,
        ...     receiver_calibration_dict=cal_dir / "receivers",
        ...     sensor_calibration_dict=cal_dir / "sensors",
        ...     save_path="output"
        ... )

        With explicit filename and HDF5 compression:

        >>> mth5_path = MakeMTH5.from_phoenix(
        ...     "data/phoenix_survey",
        ...     mth5_filename="MT_survey_2020.h5",
        ...     sample_rates=[150, 24000],
        ...     receiver_calibration_dict="calibrations/receivers",
        ...     sensor_calibration_dict="calibrations/sensors",
        ...     save_path="output",
        ...     h5_compression="gzip",
        ...     h5_compression_opts=4
        ... )

        Using explicit calibration dictionaries:

        >>> receiver_cal = {
        ...     'RX001': Path('cal/rx001_cal.json'),
        ...     'RX002': Path('cal/rx002_cal.json')
        ... }
        >>> sensor_cal = {
        ...     'SN123': phoenix_cal_obj_1,
        ...     'SN124': phoenix_cal_obj_2
        ... }
        >>> mth5_path = MakeMTH5.from_phoenix(
        ...     "data/phoenix_survey",
        ...     receiver_calibration_dict=receiver_cal,
        ...     sensor_calibration_dict=sensor_cal
        ... )

        See Also
        --------
        mt_io.phoenix.PhoenixClient : Phoenix data reader
        mt_io.phoenix.PhoenixCalibration : Calibration file handler
        """

        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        phx_client = PhoenixClient(
            data_path,
            mth5_filename=mth5_filename,
            sample_rates=sample_rates,
            receiver_calibration_dict=receiver_calibration_dict,
            sensor_calibration_dict=sensor_calibration_dict,
            save_path=save_path,
            **kw_dict,
        )

        return phx_client.make_mth5_from_phoenix()

    @classmethod
    def from_lemi424(
        cls,
        data_path: str | Path,
        survey_id: str,
        station_id: str,
        mth5_filename: str = "from_lemi424.h5",
        save_path: str | Path = Path().cwd(),
        **kwargs,
    ):
        """
        Create MTH5 from LEMI-424 long period data.

        Builds an MTH5 file from LEMI-424 instrument data on a station-by-station
        basis. LEMI data has limited metadata, so survey and station IDs must
        be provided.

        Parameters
        ----------
        data_path : str or Path
            Directory where LEMI-424 data files are stored. Can be single
            station or full directory.
        survey_id : str
            Survey ID to apply to all stations
        station_id : str
            Station ID for this station's data
        mth5_filename : str, default 'from_lemi424.h5'
            Filename for the MTH5 output file
        save_path : str or Path, default current directory
            Directory to save MTH5 file
        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip').

        Returns
        -------
        Path
            Path to created MTH5 file

        Notes
        -----
        LEMI-424 is a long-period magnetotelluric instrument. Data files have
        limited embedded metadata, requiring manual specification of survey
        and station information.

        Process each station individually due to minimal automatic metadata
        extraction capabilities.

        Examples
        --------
        Create MTH5 from LEMI-424 data:

        >>> from mth5.clients import MakeMTH5
        >>> from pathlib import Path
        >>>
        >>> data_dir = Path("data/lemi_mt01")
        >>> mth5_path = MakeMTH5.from_lemi424(
        ...     data_dir,
        ...     survey_id='MT2020',
        ...     station_id='MT01',
        ...     save_path="output"
        ... )

        With HDF5 compression:

        >>> mth5_path = MakeMTH5.from_lemi424(
        ...     "data/lemi_mt01",
        ...     survey_id='MT2020',
        ...     station_id='MT01',
        ...     mth5_filename='MT2020_MT01.h5',
        ...     h5_compression='gzip',
        ...     h5_compression_opts=1
        ... )

        Multiple stations (process individually):

        >>> for station in ['MT01', 'MT02', 'MT03']:
        ...     data_dir = Path(f"data/lemi_{station.lower()}")
        ...     mth5_path = MakeMTH5.from_lemi424(
        ...         data_dir,
        ...         survey_id='MT2020',
        ...         station_id=station,
        ...         mth5_filename=f'MT2020_{station}.h5',
        ...         save_path="output"
        ...     )

        See Also
        --------
        mt_io.lemi424.LEMI424Client : LEMI-424 data reader
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()
        kw_dict.pop("mth5_filename", None)
        kw_dict.pop("save_path", None)

        lemi_client = LEMI424Client(
            data_path,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return lemi_client.make_mth5_from_lemi424(survey_id, station_id)

    @classmethod
    def from_metronix(
        cls,
        data_path: str | Path,
        sample_rates: list[float] = [128],
        mth5_filename: str | None = None,
        save_path: str | Path | None = None,
        run_name_zeros: int = 0,
        **kwargs,
    ):
        """
        Create MTH5 from Metronix Geophysics ATSS + JSON files.

        Builds an MTH5 file from Metronix data in their new folder structure
        format with ATSS time series and JSON metadata files.

        Parameters
        ----------
        data_path : str or Path
            Highest level directory to archive data from, usually the survey
            level. For single station, use station folder path.
        sample_rates : list of float, default [128]
            Sample rates to archive in samples/second
        mth5_filename : str, optional
            Filename for the MTH5 file. If None, automatically generated from
            survey/station information.
        save_path : str or Path, optional
            Directory to save MTH5 file. If None, saves to current working
            directory.
        run_name_zeros : int, default 0
            Number of zeros for zero-padding in run names. Run names formatted
            as 'sr{sample_rate}_{run_id:0{run_name_zeros}}'. If 0, uses
            original run names (e.g., 'run_0001').
        **kwargs : dict
            Additional keyword arguments. HDF5 parameters should be prefixed
            with 'h5_' (e.g., h5_compression='gzip').

        Returns
        -------
        Path
            Path to created MTH5 file

        Notes
        -----
        Metronix Geophysics uses a specific folder structure with ATSS binary
        files and JSON metadata. The reader processes this structure and
        organizes data by survey, station, and run.

        Run naming can be customized with run_name_zeros:
        - run_name_zeros=0: Keep original names like 'run_0001'
        - run_name_zeros=4: Format as 'sr128_0001'
        - run_name_zeros=2: Format as 'sr128_01'

        Examples
        --------
        Create MTH5 from Metronix survey data:

        >>> from mth5.clients import MakeMTH5
        >>> from pathlib import Path
        >>>
        >>> data_dir = Path("data/metronix_survey")
        >>> mth5_path = MakeMTH5.from_metronix(
        ...     data_dir,
        ...     sample_rates=[128, 4096],
        ...     save_path="output"
        ... )

        With custom run naming and compression:

        >>> mth5_path = MakeMTH5.from_metronix(
        ...     "data/metronix_survey",
        ...     sample_rates=[128],
        ...     mth5_filename="survey_2020.h5",
        ...     run_name_zeros=4,
        ...     h5_compression="gzip",
        ...     h5_compression_opts=4
        ... )

        Single station with original run names:

        >>> mth5_path = MakeMTH5.from_metronix(
        ...     "data/metronix_survey/Station_001",
        ...     sample_rates=[128, 512],
        ...     run_name_zeros=0,
        ...     save_path="output"
        ... )

        Multiple sample rates with formatted names:

        >>> mth5_path = MakeMTH5.from_metronix(
        ...     "data/metronix_survey",
        ...     sample_rates=[32, 128, 512, 4096],
        ...     run_name_zeros=3,
        ...     mth5_filename="mt_data.h5"
        ... )

        See Also
        --------
        mt_io.metronix.MetronixClient : Metronix data reader
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        metronix_client = MetronixClient(
            data_path,
            sample_rates=sample_rates,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return metronix_client.make_mth5_from_metronix(run_name_zeros=run_name_zeros)

    @classmethod
    def from_nims(
        cls,
        data_path,
        sample_rates=[4096, 1024, 256],
        save_path=None,
        calibration_path=None,
        survey_id=None,
        combine=True,
        **kwargs,
    ):
        """
        Create an MTH5 from nims data.

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_nims(
            data_path, **{'h5_compression_opts': 1}
            )

        :param data_path: directory to where data are stored
        :type data_path: Path, str
        :param sample_rates: sample rates to include,
         defaults to [4096, 1024, 256]
        :type sample_rates: list, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :param calibration_path: path to calibration file amtant.cal,
         defaults to None
        :type calibration_path: str or Path, optional
        :param survey_id: survey ID to apply to all station found under
         `data_path`, defaults to None
        :type survey_id: string
        :param combine: if True combine the runs into a single run sampled at 1s,
         defaults to True
        :type combine: bool
        :return: MTH5 file name
        :rtype: Path

        """

        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        nc = NIMSClient(
            data_path,
            sample_rates=sample_rates,
            save_path=save_path,
            calibration_path=calibration_path,
            **kw_dict,
        )

        return nc.make_mth5_from_nims(survey_id=survey_id, combine=combine, **kwargs)
