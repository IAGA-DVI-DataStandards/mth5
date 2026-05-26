# -*- coding: utf-8 -*-
from __future__ import annotations


"""Transfer function HDF5 helpers for MTH5."""

from typing import Any, Iterable

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import xarray as xr

from mth5.groups import BaseGroup, EstimateDataset
from mth5.helpers import from_numpy_type, validate_name
from mth5.utils.exceptions import MTH5Error


def _check_channel_in_output(
    output_channels: Iterable[str] | None, channel: str
) -> bool:
    """Return ``True`` if ``channel`` is present in an output list.

    Handles both normal lists and corrupted serialization from HDF5 attributes
    (for example ``['"ex"', '"ey"']``).

    Parameters
    ----------
    output_channels : Iterable[str] or None
        Output channel names, potentially serialized oddly in HDF5 attributes.
    channel : str
        Channel name to search for.

    Returns
    -------
    bool
        ``True`` when the channel is detected, otherwise ``False``.

    Examples
    --------
    >>> _check_channel_in_output(["ex", "ey"], "ex")
    True
    >>> _check_channel_in_output(['"ex"', '"ey"'], "ex")
    True
    >>> _check_channel_in_output([], "hx")
    False
    """
    if not output_channels:
        return False

    # Handle normal case
    if channel in output_channels:
        return True

    # Handle corrupted HDF5 attribute serialization case
    # where ['ex', 'ey', 'hz'] becomes ['["ex"', '"ey"', '"hz"]']
    for item in output_channels:
        if isinstance(item, str):
            # Check if the channel appears in the corrupted string
            if f'"{channel}"' in item or f"'{channel}'" in item:
                return True
                # Also check for cases where the quotes are missing
            if channel in item:
                return True

    return False


from mt_metadata.timeseries import Electric, Magnetic, Run
from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate


# =============================================================================
# Transfer Functions Group
# =============================================================================
class TransferFunctionsGroup(BaseGroup):
    """Container for transfer functions under a station.

    Each child group is a single transfer function estimation managed by
    :class:`TransferFunctionGroup`.

    Examples
    --------
    >>> from mth5 import mth5
    >>> m5 = mth5.MTH5()
    >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
    >>> station = m5.stations_group.add_station("mt01")
    >>> tf_group = station.transfer_functions_group
    >>> tf_group.groups_list
    []
    """

    def __init__(self, group: Any, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)

    def tf_summary(self, as_dataframe: bool = True) -> pd.DataFrame | np.ndarray:
        """Summarize transfer functions stored for the station.

        Parameters
        ----------
        as_dataframe : bool, default True
            If ``True`` return a pandas DataFrame, otherwise a NumPy structured array.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            Summary rows including station reference, location, and TF metadata.

        Examples
        --------
        >>> summary = tf_group.tf_summary()
        >>> summary.columns[:4].tolist()  # doctest: +SKIP
        ['station_hdf5_reference', 'station', 'latitude', 'longitude']
        """

        tf_list = []
        for tf_id in self.groups_list:
            tf_group = self.get_transfer_function(tf_id)
            tf_entry = tf_group.tf_entry

            tf_entry["station_hdf5_reference"][:] = self.hdf5_group.parent.ref
            tf_entry["station"][:] = self.hdf5_group.parent.attrs["id"]
            tf_entry["latitude"][:] = self.hdf5_group.parent.attrs["location.latitude"]
            tf_entry["longitude"][:] = self.hdf5_group.parent.attrs[
                "location.longitude"
            ]
            tf_entry["elevation"][:] = self.hdf5_group.parent.attrs[
                "location.elevation"
            ]

            tf_list.append(tf_entry)
        tf_list = np.array(tf_list)

        if as_dataframe:
            return pd.DataFrame(tf_list.flatten())
        return tf_list

    def _update_time_period_from_tf(self, tf_object: TF) -> None:
        """Propagate run time bounds from a TF object into station metadata."""

        if "1980" not in tf_object.station_metadata.time_period.start:
            if "1980" in self.hdf5_group.parent.attrs["time_period.start"]:
                self.hdf5_group.parent.attrs[
                    "time_period.start"
                ] = tf_object.station_metadata.time_period.start.isoformat()

            elif (
                self.hdf5_group.parent.attrs["time_period.start"]
                != tf_object.station_metadata.time_period.start
            ):
                if (
                    self.hdf5_group.parent.attrs["time_period.start"]
                    > tf_object.station_metadata.time_period.start
                ):
                    self.hdf5_group.parent.attrs[
                        "time_period.start"
                    ] = tf_object.station_metadata.time_period.start.isoformat()

        if "1980" not in tf_object.station_metadata.time_period.end:
            if "1980" in self.hdf5_group.parent.attrs["time_period.end"]:
                self.hdf5_group.parent.attrs[
                    "time_period.end"
                ] = tf_object.station_metadata.time_period.end.isoformat()

            elif (
                self.hdf5_group.parent.attrs["time_period.end"]
                != tf_object.station_metadata.time_period.end
            ):
                if (
                    self.hdf5_group.parent.attrs["time_period.end"]
                    > tf_object.station_metadata.time_period.end
                ):
                    self.hdf5_group.parent.attrs[
                        "time_period.end"
                    ] = tf_object.station_metadata.time_period.end.isoformat()

    def add_transfer_function(
        self, name: str, tf_object: TF | None = None
    ) -> "TransferFunctionGroup":
        """Add a transfer function group under this station.

        Parameters
        ----------
        name : str
            Transfer function identifier.
        tf_object : TF, optional
            Transfer function instance to seed metadata and datasets.

        Returns
        -------
        TransferFunctionGroup
            Wrapper for the created or existing transfer function.

        Examples
        --------
        >>> tf_group = station.transfer_functions_group
        >>> _ = tf_group.add_transfer_function("mt01_4096")
        """
        name = validate_name(name)

        if tf_object is not None:
            self._update_time_period_from_tf(tf_object)
            tf_group = TransferFunctionGroup(
                self.hdf5_group.create_group(name),
                group_metadata=tf_object.station_metadata.transfer_function,
                **self.dataset_options,
            )
            tf_group.from_tf_object(tf_object, update_metadata=False)

        else:
            tf_group = TransferFunctionGroup(
                self.hdf5_group.create_group(name), **self.dataset_options
            )

        return tf_group

    def get_transfer_function(self, tf_id: str) -> "TransferFunctionGroup":
        """Return an existing transfer function by id.

        Parameters
        ----------
        tf_id : str
            Name of the transfer function.

        Returns
        -------
        TransferFunctionGroup
            Wrapper for the requested transfer function.

        Raises
        ------
        MTH5Error
            If the transfer function does not exist.

        Examples
        --------
        >>> existing = station.transfer_functions_group.get_transfer_function("mt01_4096")
        >>> existing.name  # doctest: +SKIP
        'mt01_4096'
        """

        tf_id = validate_name(tf_id)
        try:
            return TransferFunctionGroup(self.hdf5_group[tf_id], **self.dataset_options)
        except KeyError:
            msg = f"{tf_id} does not exist, " + "check station_list for existing names"
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)

    def remove_transfer_function(self, tf_id: str) -> None:
        """Delete a transfer function reference from the station.

        Parameters
        ----------
        tf_id : str
            Transfer function name.

        Notes
        -----
        HDF5 deletion removes the reference only; storage is not reclaimed.

        Examples
        --------
        >>> tf_group.remove_transfer_function("mt01_4096")
        """

        tf_id = validate_name(tf_id)
        try:
            del self.hdf5_group[tf_id]
            self.logger.info(
                "Deleting a station does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = f"{tf_id} does not exist, " "check station_list for existing names"
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)

    def get_tf_object(self, tf_id: str) -> TF:
        """Return a populated :class:`mt_metadata.transfer_functions.core.TF`.

        Parameters
        ----------
        tf_id : str
            Transfer function name to convert.

        Returns
        -------
        mt_metadata.transfer_functions.core.TF
            Transfer function populated with metadata and estimates.

        Examples
        --------
        >>> tf_obj = tf_group.get_tf_object("mt01_4096")  # doctest: +SKIP
        """

        tf_group = self.get_transfer_function(tf_id)

        return tf_group.to_tf_object()


class TransferFunctionGroup(BaseGroup):
    """Wrapper for a single transfer function estimation."""

    def __init__(self, group: Any, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)

        self._accepted_estimates = [
            "transfer_function",
            "transfer_function_error",
            "inverse_signal_power",
            "residual_covariance",
            "impedance",
            "impedance_error",
            "tipper",
            "tipper_error",
        ]

        self._period_metadata = StatisticalEstimate(
            **{
                "name": "period",
                "data_type": "real",
                "description": "Periods at which transfer function is estimated",
                "units": "samples per second",
            }
        )

    def has_estimate(self, estimate: str) -> bool:
        """Return ``True`` if an estimate exists and is populated."""

        if estimate in self.groups_list:
            est = self.get_estimate(estimate)
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            return True
        elif estimate in ["impedance"]:
            est = self.get_estimate("transfer_function")
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            elif _check_channel_in_output(
                est.metadata.output_channels, "ex"
            ) and _check_channel_in_output(est.metadata.output_channels, "ey"):
                return True
            return False
        elif estimate in ["tipper"]:
            est = self.get_estimate("transfer_function")
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            elif _check_channel_in_output(est.metadata.output_channels, "hz"):
                return True
            return False
        elif estimate in ["covariance"]:
            try:
                res = self.get_estimate("residual_covariance")
                isp = self.get_estimate("inverse_signal_power")

                if res.hdf5_dataset.shape != (
                    1,
                    1,
                    1,
                ) and isp.hdf5_dataset.shape != (
                    1,
                    1,
                    1,
                ):
                    return True
                return False
            except (KeyError, MTH5Error):
                return False
        return False

    @property
    def period(self) -> np.ndarray | None:
        """Return period array stored in ``period`` dataset, if present."""

        try:
            return self.hdf5_group["period"][()]
        except KeyError:
            return None

    @period.setter
    def period(self, period: Any) -> None:
        if period is not None:
            period = np.array(period, dtype=float)

            try:
                _ = self.add_statistical_estimate(
                    "period",
                    estimate_data=period,
                    estimate_metadata=self._period_metadata,
                    chunks=True,
                    max_shape=(None,),
                )
            except (OSError, RuntimeError, ValueError):
                self.logger.debug("period already exists, overwriting")
                self.hdf5_group["period"][...] = period

    def add_statistical_estimate(
        self,
        estimate_name: str,
        estimate_data: np.ndarray | xr.DataArray | None = None,
        estimate_metadata: StatisticalEstimate | None = None,
        max_shape: tuple[int | None, int | None, int | None] = (None, None, None),
        chunks: bool = True,
        **kwargs: Any,
    ) -> EstimateDataset:
        """Add a statistical estimate dataset.

        Parameters
        ----------
        estimate_name : str
            Dataset name.
        estimate_data : numpy.ndarray or xarray.DataArray, optional
            Estimate values; if ``None`` a placeholder array is created.
        estimate_metadata : StatisticalEstimate, optional
            Metadata describing the estimate.
        max_shape : tuple of int or None, default (None, None, None)
            Maximum shape for resizable datasets.
        chunks : bool, default True
            Chunking flag forwarded to HDF5 dataset creation.

        Returns
        -------
        EstimateDataset
            Wrapper combining dataset and metadata.

        Raises
        ------
        TypeError
            If ``estimate_data`` is not array-like.

        Examples
        --------
        >>> est = tf_group.add_statistical_estimate("transfer_function")
        >>> isinstance(est, EstimateDataset)
        True
        """

        estimate_name = validate_name(estimate_name)

        if estimate_metadata is None:
            estimate_metadata = StatisticalEstimate()
            estimate_metadata.name = estimate_name
        if estimate_data is not None:
            if not isinstance(estimate_data, (np.ndarray, xr.DataArray)):
                msg = f"Need to input a numpy or xarray.DataArray not {type(estimate_data)}"
                self.logger.exception(msg)
                raise TypeError(msg)
            if isinstance(estimate_data, xr.DataArray):
                estimate_metadata.output_channels = estimate_data.coords[
                    "output"
                ].values.tolist()
                estimate_metadata.input_channels = estimate_data.coords[
                    "input"
                ].values.tolist()
                estimate_metadata.name = validate_name(estimate_data.name)
                estimate_metadata.data_type = estimate_data.dtype.name

                estimate_data = estimate_data.to_numpy()
            dtype = estimate_data.dtype
        else:
            dtype = complex
            chunks = True
            estimate_data = np.zeros((1, 1, 1), dtype=dtype)
        try:
            dataset = self.hdf5_group.create_dataset(
                estimate_name,
                data=estimate_data,
                dtype=dtype,
                chunks=chunks,
                maxshape=max_shape,
                **self.dataset_options,
            )

            estimate_dataset = EstimateDataset(
                dataset, dataset_metadata=estimate_metadata
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
            msg = f"estimate {estimate_metadata.name} already exists, returning existing group."
            self.logger.debug(msg)

            estimate_dataset = self.get_estimate(estimate_metadata.name)
        return estimate_dataset

    def get_estimate(self, estimate_name: str) -> EstimateDataset:
        """Return a statistical estimate dataset by name."""
        estimate_name = validate_name(estimate_name)

        try:
            estimate_dataset = self.hdf5_group[estimate_name]
            estimate_metadata = StatisticalEstimate(**dict(estimate_dataset.attrs))
            return EstimateDataset(
                estimate_dataset,
                dataset_metadata=estimate_metadata,
                write_metadata=False,
            )
        except KeyError:
            msg = (
                f"{estimate_name} does not exist, "
                "check groups_list for existing names"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)
        except OSError as error:
            self.logger.error(error)
            raise MTH5Error(error)

    def remove_estimate(self, estimate_name: str) -> None:
        """Remove a statistical estimate dataset reference."""

        estimate_name = validate_name(estimate_name.lower())

        try:
            del self.hdf5_group[estimate_name]
            self.logger.info(
                "Deleting a estimate does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{estimate_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)

    def to_tf_object(self) -> TF:
        """Convert this group into a populated :class:`TF` object.

        Returns
        -------
        mt_metadata.transfer_functions.core.TF
            TF instance with survey, station, runs, channels, period, and
            estimate datasets applied.

        Raises
        ------
        ValueError
            If no period dataset is present.

        Examples
        --------
        >>> tf_obj = tf_group.to_tf_object()  # doctest: +SKIP
        """

        tf_obj = TF()

        # get survey metadata
        survey_dict = dict(self.hdf5_group.parent.parent.parent.parent.attrs)
        for key, value in survey_dict.items():
            survey_dict[key] = from_numpy_type(value)
        tf_obj.survey_metadata.from_dict({"survey": survey_dict})

        # get station metadata
        station_dict = dict(self.hdf5_group.parent.parent.attrs)
        for key, value in station_dict.items():
            station_dict[key] = from_numpy_type(value)
        tf_obj.station_metadata.from_dict({"station": station_dict})

        # need to update transfer function metadata
        tf_dict = dict(self.hdf5_group.attrs)
        for key, value in tf_dict.items():
            tf_dict[key] = from_numpy_type(value)
        tf_obj.station_metadata.transfer_function.from_dict(
            {"transfer_function": tf_dict}
        )

        # add run and channel metadata
        tf_obj.station_metadata.runs = []
        for run_id in tf_obj.station_metadata.transfer_function.runs_processed:
            if run_id in ["", None, "None"]:
                continue
            try:
                run = self.hdf5_group.parent.parent[validate_name(run_id)]
                run_dict = dict(run.attrs)
                for key, value in run_dict.items():
                    run_dict[key] = from_numpy_type(value)
                run_obj = Run(**run_dict)

                for ch_id in run.keys():
                    ch = run[validate_name(ch_id)]
                    ch_dict = dict(ch.attrs)
                    for key, value in ch_dict.items():
                        ch_dict[key] = from_numpy_type(value)
                    if ch_dict["type"] == "electric":
                        ch_obj = Electric(**ch_dict)
                    elif ch_dict["type"] == "magnetic":
                        ch_obj = Magnetic(**ch_dict)
                    run_obj.add_channel(ch_obj)
                tf_obj.station_metadata.add_run(run_obj)
            except KeyError:
                self.logger.info(f"Could not get run {run_id} for transfer function")
        if self.period is not None:
            tf_obj.period = self.period
        else:
            msg = "Period must not be None to create a transfer function object"
            self.logger.error(msg)
            raise ValueError(msg)
        for estimate_name in self.groups_list:
            if estimate_name in ["period"]:
                continue
            estimate = self.get_estimate(estimate_name)

            try:
                setattr(tf_obj, estimate_name, estimate.to_numpy())
            except AttributeError as error:
                self.logger.exception(error)

        # need to update time periods
        tf_obj.station_metadata.update_time_period()
        tf_obj.survey_metadata.update_time_period()
        return tf_obj

    def from_tf_object(self, tf_obj: TF, update_metadata: bool = True) -> None:
        """Populate datasets from a :class:`TF` object.

        Parameters
        ----------
        tf_obj : TF
            Transfer function object containing estimates and metadata.
        update_metadata : bool, default True
            If ``True`` write transfer function metadata to HDF5.

        Raises
        ------
        ValueError
            If ``tf_obj`` is not a ``TF`` instance.

        Examples
        --------
        >>> tf_group.from_tf_object(tf_obj)  # doctest: +SKIP
        """

        if not isinstance(tf_obj, TF):
            msg = f"Input must be a TF object not {type(tf_obj)}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.period = tf_obj.period
        if update_metadata:
            self.metadata.update(tf_obj.station_metadata.transfer_function)
            self.write_metadata()

        # if transfer function is available then impedance and tipper are
        # redundant.
        if tf_obj.has_transfer_function():
            accepted_estimates = self._accepted_estimates[0:4]
        else:
            accepted_estimates = self._accepted_estimates
        for estimate_name in accepted_estimates:
            try:
                estimate = getattr(tf_obj, estimate_name)
                if estimate is not None:
                    _ = self.add_statistical_estimate(estimate_name, estimate)
                else:
                    self.logger.debug(f"Did not find {estimate_name} in TF. Skipping")
            except AttributeError:
                self.logger.debug(f"Did not find {estimate_name} in TF. Skipping")
