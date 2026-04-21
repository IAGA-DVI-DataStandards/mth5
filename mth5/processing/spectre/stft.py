"""
This module has methods for applying the short-time-Fourier-transform.


"""

from typing import Union

import numpy as np
import scipy.signal as ssig
import xarray as xr
from mt_metadata.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.processing.fourier_coefficients import Decimation as FCDecimation
from mt_timeseries.spectre.spectrogram import Spectrogram

from .prewhitening import apply_prewhitening, apply_recoloring


def run_ts_to_stft_scipy(
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation],
    run_xrds_orig: xr.Dataset,
) -> Spectrogram:
    """
    Converts a runts object into a time series of Fourier coefficients.
    This method uses scipy.signal.spectrogram.

    TODO: consider making this a method of RunTS; runts.to_spectrogram(decimation_obj)

    Parameters
    ----------
    decimation_obj :  Union[AuroraDecimationLevel, FCDecimation]
        Information about how the decimation level is to be processed
    run_xrds_orig : : xarray.core.dataset.Dataset
        Time series to be processed

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients
    """
    run_xrds = apply_prewhitening(decimation_obj.stft.prewhitening_type, run_xrds_orig)

    stft_obj = xr.Dataset()
    for channel_id in run_xrds.data_vars:
        ff, tt, specgm = ssig.spectrogram(
            run_xrds[channel_id].data,
            fs=decimation_obj.decimation.sample_rate,
            window=decimation_obj.stft.window.taper(),
            nperseg=decimation_obj.stft.window.num_samples,
            noverlap=decimation_obj.stft.window.overlap,
            detrend="linear",
            scaling="density",
            mode="complex",
        )

        # drop Nyquist
        ff = ff[:-1]
        specgm = specgm[:-1, :]
        specgm *= np.sqrt(
            2
        )  # compensate energy for keeping only positive harmonics (keep PSDs accurate)

        # make time_axis
        tt = tt - tt[0]
        tt *= decimation_obj.decimation.sample_rate
        time_axis = run_xrds.time.data[tt.astype(int)]

        xrd = xr.DataArray(
            specgm.T,
            dims=["time", "frequency"],
            coords={"frequency": ff, "time": time_axis},
        )
        stft_obj.update({channel_id: xrd})

    if decimation_obj.stft.recoloring:
        stft_obj = apply_recoloring(decimation_obj.stft.prewhitening_type, stft_obj)

    return Spectrogram(dataset=stft_obj)
