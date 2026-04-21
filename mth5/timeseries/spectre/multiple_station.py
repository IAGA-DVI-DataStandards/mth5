"""
Work In progress

This module is concerned with working with Fourier coefficient data

TODO:
2. Give MultivariateDataset a covariance() method

Tools include prototypes for
- extracting portions of an FC Run Time Series
- merging multiple stations runs together into an xarray
- relabelling channels to avoid namespace clashes for multi-station data

"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mt_timeseries.spectre.spectrogram import Spectrogram

import mth5.mth5
from mth5.utils.exceptions import MTH5Error


@dataclass
class FCRunChunk:
    """

    This class formalizes the required metadata to specify a chunk of a timeseries of Fourier coefficients.

    This may move to mt_metadata -- for now just use a dataclass as a prototype.
    """

    survey_id: str = "none"
    station_id: str = ""
    run_id: str = ""
    decimation_level_id: str = "0"
    start: str = ""
    end: str = ""
    channels: Tuple[str] = ()

    @property
    def start_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    @property
    def duration(self) -> pd.Timestamp:
        return self.end_timestamp - self.start_timestamp


@dataclass
class MultivariateLabelScheme:
    """
    Class to store information about how a multivariate (MV) dataset will be lablelled.

    Has a scheme to handle the how channels will be named.

    This is just a place holder to manage possible future complexity.

    It seemed like a good idea to formalize the fact that we take, by default
    f"{station}_{component}" as the MV channel label.
    It also seemed like a good idea to record what the join character is.
    In the event that we wind up with station names that have underscores in them, then we could,
    for example, set the join character to "__".

    TODO: Consider rename default to ("station", "data_var") instead of ("station", "component")

    Parameters
    ----------
    :type label_elements: tuple
    :param label_elements: This is meant to tell what information is being concatenated into an MV channel label.
    :type join_char: str
    :param join_char: The string that is used to join the label elements.

    """

    label_elements: tuple = (
        "station",
        "component",
    )
    join_char: str = "_"

    @property
    def id(self) -> str:
        return self.join(self.label_elements)

    def join(self, elements: Union[list, tuple]) -> str:
        """

        Join the label elements to a string

        :type elements:  tuple
        :param elements: Expected to be the label elements, default are (station, component)

        :return: The name of the channel (in a multiple-station context).
        :rtype: str

        """
        return self.join_char.join(elements)

    def split(self, mv_channel_name) -> dict:
        """

        Splits a multi-station channel name and returns a dict of strings, keyed by self.label_elements.
        This method is basically the reverse of self.join

        :param mv_channel_name: a multivariate channel name string
        :type mv_channel_name: str
        :return: Channel name as a dictionary.
        :rtype: dict

        """
        splitted = mv_channel_name.split(self.join_char)
        if len(splitted) != len(self.label_elements):
            msg = f"Incompatable map {splitted} and {self.label_elements}"
            logger.error(msg)
            msg = f"cannot map {len(splitted)} to {len(self.label_elements)}"
            raise ValueError(msg)
        output = dict(zip(self.label_elements, splitted))
        return output


class MultivariateDataset(Spectrogram):
    """
    Here is a container for a multivariate spectral dataset.
    The xarray is the main underlying item, but it will be useful to have functions that, for example returns a
    list of the associated stations, or that return a list of channels that are associated with a station, etc.

    This is intended to be used as a multivariate spectral dotaset at one frequency band.

    TODO: Consider making this an extension of Spectrogram
    TODO: Rename this class to MultivariateSpectrogram.


    """

    def __init__(
        self,
        dataset: xr.Dataset,
        label_scheme: Optional[MultivariateLabelScheme] = None,
    ):
        super().__init__(dataset=dataset)
        self._label_scheme = label_scheme

        self._channels = None
        self._stations = None
        self._station_channels = None

    @property
    def label_scheme(self) -> MultivariateLabelScheme:
        if self._label_scheme is None:
            msg = f"No label scheme found for {self.__class__} -- setting to default"
            logger.warning(msg)
            self._label_scheme = MultivariateLabelScheme()
        return self._label_scheme

    @property
    def channels(self) -> list:
        """
        returns a list of channels in the dataarray
        """
        if self._channels is None:
            self._channels = list(self.dataarray.coords["variable"].values)
        return self._channels

    @property
    def num_channels(self) -> int:
        """returns a count of the total number of channels in the dataset"""
        return len(self.channels)

    @property
    def stations(self) -> List[str]:
        """
        Parses the channel names, extracts the station names

        return a unique list of stations preserving order.
        """
        if self._stations is None:
            if self.label_scheme.id == "station_component":
                tmp = [self.label_scheme.split(x)["station"] for x in self.channels]
                # tmp = [x.split("_")[0] for x in self.channels]
                stations = list(dict.fromkeys(tmp))  # order preserving unique values
                self._stations = stations
            else:
                msg = f"No rule for parsting station names from label scheme {self.label_scheme.id}"
                raise NotImplementedError(msg)

        return self._stations

    def station_channels(
        self,
        station: str,
    ) -> List[str]:
        """
        This is a utility function that provides a way to access channel_names in a multivariate array associated
         with a particular station.
        The list is accessed via the self._station_channels attr, which gets set here if it has not
         been initialized previously.  self._station_channels is a dict keyed by station_id, with value
         is a list of channel names for that station.

        :param station: The name of the station.
        :type station: str

        :rtype: List[str]
        :returns: list of channel names for the input station.

        """
        # set self._station_channels is not already done
        if self._station_channels is None:
            station_channels = {}
            for station_id in self.stations:
                station_channels[station_id] = self._get_station_channel_names(
                    station_id,
                    multivariate_labels=True,
                )
            self._station_channels = station_channels

        return self._station_channels[station]

    def _get_station_channel_names(
        self, station: str, multivariate_labels: bool = True
    ) -> List[str]:
        """

        This is a utility function that to get all channel names in a multivariate array associated
         with a particular station.

        :param station: The name of the station.
        :type station: str
        :param multivariate_labels: When set to true, returned values have the "full multivariate" channel names,
        e.g. station "mt1" may return for example "mt1_ex", "mt1_ey", "mt1_hx" ... etc.  If set to false the names
        will be returned within the context of a station, so they may be for example "ex", "ey", "hx" ... etc.
        The default value is True.
        :type multivariate_labels: bool

        :rtype: List[str]
        :returns: Channel names for the input station.

        """
        station_channels = [
            x
            for x in self.channels
            if station == x.split(self.label_scheme.join_char)[0]
        ]
        if not multivariate_labels:
            station_channels = [
                x.split(self.label_scheme.join_char)[1] for x in station_channels
            ]

        return station_channels

    def archive_cross_powers(
        self,
        tf_station: str,
        with_fcs: bool = True,
    ):
        """
        tf_station: str
         This tells us under which station we should store the output of this function.
         TODO: Consider moving this to another function which performs archiving in future.

        with_fcs: bool
         If True, the features are packed into the same hdf5-group as the FCs,
         as its own dataset.
         If False: the features are packed into the hdf5 features-group.

        Returns
        -------

        """

    # TODO: Replace with Spectrogram's covariance_matrix
    def cross_power(
        self, aweights: Optional[np.ndarray] = None, bias: Optional[bool] = True
    ) -> xr.DataArray:
        """
        Calculate the cross-power from a multivariate, complex-valued array of Fourier coefficients.

        For a multivaraiate FC Dataset with n_time time windows, this returns an array with the same number of time
        windows.  At each time _t_, the result is a covariance matrix.

        Caveats and Notes:
          - This method calls numpy.cov, which means that the cross-power is computes as X@XH (rather than
          XH@X). Sometimes X*XH is referred to as the Vozoff convention, whereas XH*X could be the
          Bendat & Piersol convention.
          - np.cov subtracts the meas before computing the cross terms.
          - This methos will use the entire band of the spectrogram.

        :param X: Multivariate time series as an xarray
        :type X: xr.DataArray
        :param aweights: This is a "passthrough" parameter to numpy.cov These relative weights are typically large for
         observations considered "important" and smaller for observations considered less "important". If ``ddof=0``
         the array of weights can be used to assign probabilities to observation vectors.
        :type aweights: Optional[np.ndarray]
        :param bias: bias=True normalizes by N instead of (N-1).
        :type bias: bool

        :rtype: xr.DataArray
        :return: The covariance matrix of the data in xarray form.

        """
        X = self.dataarray
        channels = list(X.coords["variable"].values)

        S = xr.DataArray(
            np.cov(X, aweights=aweights, bias=bias),
            dims=["channel_1", "channel_2"],
            coords={"channel_1": channels, "channel_2": channels},
        )
        return S


# Weights vs masks


def calculate_mask_from_feature(
    feature_series,
    threshold_obj,  # has lower/upper bound, can be -inf, inf
):
    """

    Returns
    -------

    """
    mask1 = feature_series < threshold_obj.lower_bound
    mask2 = feature_series > threshold_obj.upper_bound
    return mask1 & mask2


def calculate_weight_from_feature(
    feature_series,
    threshold_obj,  # has lower/upper bound, can be -inf, inf
):
    """
        This calculates a weighting function based on the thresholds
        and possibly some other info, such as the distribution of the features.

        The weigth function is interpolated over the range of the feature values
        and then evaluated at the feature values.
    Parameters
    ----------
    feature_series
    threshold_obj

    Returns
    -------

    """


def merge_masks():
    """
    calcualtes a "final mask" that is loaded and applied to the data
    input to regression
    """


def merge_weights():
    """
    calcualtes a "final mask" that is loaded and applied to the data
        input to regression
    Returns
    -------

    """


# TODO: add this method to tf-estimation right before robust regression.
def apply_masks_and_weights():
    pass


def make_multistation_spectrogram(
    m: mth5.mth5.MTH5,
    fc_run_chunks: list,
    label_scheme: Optional[MultivariateLabelScheme] = MultivariateLabelScheme(),
    rtype: Optional[Literal["xrds"]] = None,
) -> Union[xr.Dataset, MultivariateDataset]:
    """

    See notes in mth5 issue #209.  Takes a list of FCRunChunks and returns the largest contiguous
    block of multichannel FC data available.

    |----------Station 1 ------------|
            |----------Station 2 ------------|
    |--------------------Station 3 ----------------------|


            |-------RETURNED------|

    Handle additional runs in a separate call to this function and then concatenate time series afterwards.

    Input must specify N (station-run-start-end-channel_list) tuples.
    If channel_list is not provided, get all channels.
    If start-end are not provided, read the whole run -- warn if runs are not all synchronous, and
    truncate all to max(starts), min(ends) after the start and end times are sorted out.

    Station IDs must be unique.

    :param m:  The mth5 object to get the FCs from.
    :type m: mth5.mth5.MTH5
    :param fc_run_chunks: Each element of this describes a chunk of a run to load from stored FCs.
    :type fc_run_chunks: list
    :param label_scheme: Specifies how the channels are to be named in the multivariate xarray.
    :type label_scheme: Optional[MultivariateLabelScheme]
    :param rtype: Specifies whether to return an xarray or a MultivariateDataset.  Currently only supports "xrds",
    otherwise will return MultivariateDataset.
    :type rtype: Optional[Literal["xrds"]]

    :rtype: Union[xarray.Dataset, MultivariateDataset]:
    :return: The multivariate dataset, either as an xarray or as a MultivariateDataset

    """
    for i_fcrc, fcrc in enumerate(fc_run_chunks):
        station_obj = m.get_station(fcrc.station_id, fcrc.survey_id)
        station_fc_group = station_obj.fourier_coefficients_group
        try:
            run_fc_group = station_obj.fourier_coefficients_group.get_fc_group(
                fcrc.run_id
            )
        except MTH5Error as e:
            error_msg = f"Failed to get fc group {fcrc.run_id}"
            logger.error(error_msg)
            msg = f"Available FC Groups for station {fcrc.station_id}: "
            msg = f"{msg} {station_fc_group.groups_list}"
            logger.error(msg)
            logger.error(f"Maybe try adding FCs for {fcrc.run_id}")
            raise e  # MTH5Error(error_msg)

        fc_dec_level = run_fc_group.get_decimation_level(fcrc.decimation_level_id)
        if fcrc.channels:
            channels = list(fcrc.channels)
        else:
            channels = None

        fc_dec_level_xrds = fc_dec_level.to_xarray(channels=channels)
        # could create name mapper dict from run_fc_group.channel_summary here if we wanted to.

        if fcrc.start:
            # TODO: Push slicing into the to_xarray() command so we only access what we need -- See issue #212
            cond = fc_dec_level_xrds.time >= fcrc.start_timestamp
            msg = f"trimming  {sum(~cond.data)} samples to {fcrc.start} "
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")

        if fcrc.end:
            # TODO: Push slicing into the to_xarray() command so we only access what we need -- See issue #212
            cond = fc_dec_level_xrds.time <= fcrc.end_timestamp
            msg = f"trimming  {sum(~cond.data)} samples to {fcrc.end} "
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")

        if label_scheme.id == "station_component":
            name_dict = {
                f"{x}": label_scheme.join((fcrc.station_id, x))
                for x in fc_dec_level_xrds.data_vars
            }
        else:
            msg = f"Label Scheme elements {label_scheme.id} not implemented"
            raise NotImplementedError(msg)

        if i_fcrc == 0:
            xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
        else:
            fc_dec_level_xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
            xrds = xrds.merge(fc_dec_level_xrds)

    # Check that no nan came about as a result of the merge
    if bool(xrds.to_array().isnull().any()):
        msg = "Nan detected in multistation spectrogram"
        logger.warning(msg)

    if rtype == "xrds":
        output = xrds
    else:
        output = MultivariateDataset(dataset=xrds, label_scheme=label_scheme)

    return output
