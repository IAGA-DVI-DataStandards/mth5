#!/usr/bin/env python
# coding: utf-8

# # Make an MTH5 from Phoenix Data
#
# This example demonstrates how to read Phoenix data into an MTH5 file.  The data comes from example data in [PhoenixGeoPy](https://github.com/torresolmx/PhoenixGeoPy). Here I downloaded those data into a local folder on my computer by forking the main branch.

# ## Imports

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mt_io.phoenix import PhoenixCollection
from mt_io.phoenix.readers.calibrations import PhoenixCalibration
from mt_metadata.timeseries import AppliedFilter

from mth5 import read_file
from mth5.clients.base import ClientBase
from mth5.mth5 import MTH5


# =============================================================================


class PhoenixClient(ClientBase):
    def __init__(
        self,
        data_path: str | Path,
        sample_rates: list[int] = [150, 24000],
        save_path: str | Path | None = None,
        receiver_calibration_dict: dict | str | Path = {},
        sensor_calibration_dict: dict | str | Path = {},
        mth5_filename: str = "from_phoenix.h5",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=sample_rates,
            mth5_filename=mth5_filename,
            **kwargs,
        )

        self.receiver_calibration_dict = receiver_calibration_dict
        self.sensor_calibration_dict = sensor_calibration_dict

        self.collection = PhoenixCollection(self.data_path)

    @property
    def receiver_calibration_dict(self) -> dict:
        """Receiver calibrations.

        Returns
        -------
        dict
            Dictionary mapping receiver IDs to calibration file paths.

        Examples
        --------
        >>> client = PhoenixClient('data/path')
        >>> client.receiver_calibration_dict = {'RX001': Path('RX001_rxcal.json')}
        >>> client.receiver_calibration_dict
        {'RX001': Path('RX001_rxcal.json')}
        """
        return self._receiver_calibration_dict

    @receiver_calibration_dict.setter
    def receiver_calibration_dict(self, value: dict | str | Path) -> None:
        """Set receiver calibration dictionary from dict or path.

        Parameters
        ----------
        value : dict or str or Path
            Dictionary of calibrations or path to calibration files.

        Raises
        ------
        TypeError
            If value is not a dict, str, or Path.

        Examples
        --------
        >>> client = PhoenixClient('data/path')
        >>> client.receiver_calibration_dict = 'calibrations/'
        >>> client.receiver_calibration_dict  # doctest: +SKIP
        {'RX001': Path('calibrations/RX001_rxcal.json'), ...}
        """
        self._receiver_calibration_dict = {}
        if isinstance(value, dict):
            self._receiver_calibration_dict = value
        elif isinstance(value, (str, Path)):
            receiver_path = Path(value)
            if receiver_path.is_dir():
                self._receiver_calibration_dict = {}
                for fn in list(receiver_path.rglob("*.rxcal.json")) + list(
                    receiver_path.rglob("*.rx_cal.json")
                ):
                    self._receiver_calibration_dict[
                        fn.stem.split(".")[0].split("_")[0]
                    ] = fn
            elif receiver_path.is_file():
                self._receiver_calibration_dict = {}
                key = receiver_path.stem.split(".")[0].split("_")[0]
                self._receiver_calibration_dict[key] = receiver_path
        elif value is None:
            pass
        else:
            raise TypeError(f"type {type(value)} not supported.")

    @property
    def sensor_calibration_dict(self) -> dict:
        """Sensor calibration dictionary.

        Returns
        -------
        dict
            Dictionary mapping sensor IDs to PhoenixCalibration objects.

        Examples
        --------
        >>> client = PhoenixClient('data/path')
        >>> client.sensor_calibration_dict = {'H001': PhoenixCalibration('H001_scal.json')}
        >>> client.sensor_calibration_dict['H001']  # doctest: +SKIP
        <PhoenixCalibration object>
        """
        return self._sensor_calibration_dict

    @sensor_calibration_dict.setter
    def sensor_calibration_dict(self, value: dict | str | Path) -> None:
        """Set sensor calibration dictionary from dict or path.

        Parameters
        ----------
        value : dict or str or Path
            Dictionary of calibrations or path to calibration files.

        Raises
        ------
        ValueError
            If value is not a dict, str, or Path.

        Examples
        --------
        >>> client = PhoenixClient('data/path')
        >>> client.sensor_calibration_dict = 'calibrations/'
        >>> client.sensor_calibration_dict  # doctest: +SKIP
        {'H001': <PhoenixCalibration object>, ...}
        """
        if isinstance(value, dict):
            self._sensor_calibration_dict = value
        elif isinstance(value, (str, Path)):
            self._sensor_calibration_dict = {}
            cal_path = Path(value)
            if cal_path.is_dir():
                for fn in cal_path.rglob("*scal.json"):
                    self._sensor_calibration_dict[
                        fn.stem.split(".")[0].split("_")[0]
                    ] = PhoenixCalibration(fn)
            elif cal_path.is_file():
                if not cal_path.exists():
                    raise IOError(f"Could not find {cal_path}")
                key = cal_path.stem.split(".")[0].split("_")[0]
                self._sensor_calibration_dict[key] = PhoenixCalibration(cal_path)
        elif value is None:
            pass
        else:
            raise ValueError("calibration_path cannot be None")

    def make_mth5_from_phoenix(self, **kwargs: dict) -> str | Path | None:
        """Make an MTH5 from Phoenix files.

        Split into runs, account for filters. Updates the MTH5 file with Phoenix data.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments to override instance attributes.

        Returns
        -------
        str, Path, or None
            Path to the saved MTH5 file.

        Examples
        --------
        >>> client = PhoenixClient('data/path', save_path='output.h5')
        >>> client.make_mth5_from_phoenix()
        'output.h5'
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        run_dict = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, self.mth5_file_mode)

            for station_id, station_dict in run_dict.items():
                collection_metadata = self.collection.metadata_dict[station_id]
                survey_metadata = collection_metadata.survey_metadata
                survey_group = m.add_survey(survey_metadata.id)

                station_metadata = self.collection.metadata_dict[
                    station_id
                ].station_metadata
                station_group = survey_group.stations_group.add_station(
                    station_metadata.id, station_metadata=station_metadata
                )
                # get receiver calibration file if it exists
                try:
                    rxcal_fn = self.receiver_calibration_dict[
                        collection_metadata.instrument_id
                    ]
                except KeyError:
                    rxcal_fn = None
                    self.logger.warning(
                        f"Could not find receiver {collection_metadata.instrument_id} in "
                        f"receiver calibrations dict {self.receiver_calibration_dict}."
                    )
                # loop over runs for the station and add channels to the run group
                for run_id, run_df in station_dict.items():
                    run_metadata = collection_metadata.run_metadata
                    run_metadata.id = run_id
                    run_metadata.sample_rate = float(run_df.sample_rate.unique()[0])

                    run_group = station_group.add_run(
                        run_metadata.id, run_metadata=run_metadata
                    )
                    for row in run_df.itertuples():
                        try:
                            ch_ts = read_file(
                                row.fn,
                                **{
                                    "channel_map": collection_metadata.channel_map,
                                    "rxcal_fn": rxcal_fn,
                                },
                            )
                        except OSError as error:
                            self.logger.warning(
                                f"OSError: skipping {row.fn.name} likely too small.  Error: {error}"
                            )
                            continue

                        if ch_ts.component in ["h1", "h2", "h3"]:
                            # for phx coils from generic response curves
                            if (
                                ch_ts.channel_metadata.sensor.id
                                in self.sensor_calibration_dict.keys()
                            ):
                                pc = self.sensor_calibration_dict[
                                    ch_ts.channel_metadata.sensor.id
                                ]
                                for key in pc.__dict__.keys():
                                    if key.startswith("h"):
                                        break
                                coil_fap = getattr(pc, key)

                                # create applied filter
                                applied_filter = AppliedFilter(
                                    name=coil_fap.name,
                                    applied=True,
                                    stage=len(ch_ts.channel_metadata.filters),
                                )

                                # update existing filter if it exists, otherwise add it
                                ch_ts.channel_metadata.add_filter(applied_filter)

                                # update existing filter in channel response if it exists, otherwise add it
                                if coil_fap.name not in ch_ts.channel_response.names:
                                    ch_ts.channel_response.add_filter(coil_fap)
                                else:
                                    ch_ts.channel_response.replace_filter(coil_fap)

                            else:
                                self.logger.warning(
                                    f"Could not find coil {ch_ts.channel_metadata.sensor.id} in sensor calibrations."
                                )

                        # check to make sure all filters are applied True
                        # this is a hack for now.
                        for f in ch_ts.channel_metadata.filters:
                            f.applied = True

                        # add channel to the run group
                        run_group.from_channel_ts(ch_ts)

                    run_group.update_metadata()

            station_group.update_metadata()
            survey_group.update_metadata()

        return self.save_path
