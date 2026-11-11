"""
Methods for pre-whitening time series before Fourier transformation.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import xarray as xr
from loguru import logger


@dataclass(frozen=True)
class PrewhiteningResult:
    """Fitted per-channel filter coefficients used to recolor Fourier data."""

    ar_coefficients: tuple[float, ...] = ()
    ma_coefficients: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, list[float]]:
        """Return JSON-serializable filter coefficients."""
        return {
            "ar_coefficients": list(self.ar_coefficients),
            "ma_coefficients": list(self.ma_coefficients),
        }


def _prewhitening_type(stft_config) -> str:
    """Normalize metadata enum values to their serialized strings."""
    prewhitening_type = stft_config.prewhitening_type
    return getattr(prewhitening_type, "value", prewhitening_type)


def _validate_arma_configuration(stft_config) -> tuple[int, int]:
    """Validate the AR-only subset of the ARMA configuration family."""
    ar_order = stft_config.prewhitening_ar_order
    ma_order = stft_config.prewhitening_ma_order
    if ar_order < 1:
        msg = "ARMA prewhitening requires prewhitening_ar_order to be at least 1"
        raise ValueError(msg)
    if ma_order != 0:
        msg = "ARMA prewhitening currently supports only " "prewhitening_ma_order == 0"
        raise NotImplementedError(msg)
    return ar_order, ma_order


def _fit_ar_residuals(
    channel_data: np.ndarray,
    ar_order: int,
    channel_id: str,
) -> tuple[np.ndarray, PrewhiteningResult]:
    """Fit a stable AR model and return its innovation sequence.

    The model's initialized one-step residuals are retained so whitening does
    not shift the run time coordinate.
    """
    if channel_data.ndim != 1:
        msg = f"Channel {channel_id} must have one-dimensional time-series data"
        raise ValueError(msg)
    if channel_data.size <= ar_order + 1:
        msg = (
            f"Channel {channel_id} has {channel_data.size} samples, which is too "
            f"short for AR order {ar_order}"
        )
        raise ValueError(msg)
    if not np.isfinite(channel_data).all():
        msg = f"Channel {channel_id} contains missing or non-finite samples"
        raise ValueError(msg)

    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(
        channel_data,
        order=(ar_order, 0, 0),
        trend="n",
        enforce_stationarity=True,
    )
    fitted_model = model.fit()
    if np.any(np.abs(fitted_model.arroots) <= 1):
        msg = f"Channel {channel_id} produced an unstable AR filter"
        raise ValueError(msg)

    result = PrewhiteningResult(
        ar_coefficients=tuple(float(value) for value in fitted_model.arparams),
        ma_coefficients=(),
    )
    return np.asarray(fitted_model.resid), result


def apply_prewhitening(
    stft_config,
    run_xrds_input: xr.Dataset,
) -> tuple[xr.Dataset, dict[str, PrewhiteningResult]]:
    """
    Apply configured prewhitening over the full input run.

    Returns the whitened time series and fitted per-channel filter metadata. The
    metadata must be passed to :func:`apply_recoloring`; filters are never refit
    from Fourier data.
    """
    prewhitening_type = _prewhitening_type(stft_config)
    if not prewhitening_type:
        logger.info("No prewhitening specified - skipping this step")
        return run_xrds_input, {}

    if prewhitening_type == "first difference":
        return run_xrds_input.diff("time", label="upper"), {}

    if prewhitening_type != "ARMA":
        msg = f"{prewhitening_type} pre-whitening not implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    ar_order, _ = _validate_arma_configuration(stft_config)
    run_xrds = run_xrds_input.copy(deep=True)
    results: dict[str, PrewhiteningResult] = {}
    for channel_id, channel in run_xrds_input.data_vars.items():
        residuals, result = _fit_ar_residuals(
            np.asarray(channel.data),
            ar_order,
            channel_id,
        )
        run_xrds[channel_id].data = residuals
        run_xrds[channel_id].attrs["prewhitening"] = result.as_dict()
        results[channel_id] = result
    return run_xrds, results


def _whitening_response(
    prewhitening_type: str,
    frequencies: np.ndarray,
    sample_rate: float,
    result: PrewhiteningResult | None,
) -> np.ndarray:
    """Compute the exact discrete-time whitening response."""
    z_inverse = np.exp(-2.0j * np.pi * frequencies / sample_rate)
    if prewhitening_type == "first difference":
        return 1.0 - z_inverse

    if prewhitening_type == "ARMA":
        if result is None:
            msg = "ARMA recoloring requires fitted per-channel coefficients"
            raise ValueError(msg)
        ar_polynomial = np.ones_like(z_inverse, dtype=complex)
        for order, coefficient in enumerate(result.ar_coefficients, start=1):
            ar_polynomial -= coefficient * z_inverse**order

        ma_polynomial = np.ones_like(z_inverse, dtype=complex)
        for order, coefficient in enumerate(result.ma_coefficients, start=1):
            ma_polynomial += coefficient * z_inverse**order
        return ar_polynomial / ma_polynomial

    msg = f"{prewhitening_type} recoloring not implemented"
    raise NotImplementedError(msg)


def apply_recoloring(
    stft_config,
    stft_obj: xr.Dataset,
    sample_rate: float,
    results: Mapping[str, PrewhiteningResult],
) -> xr.Dataset:
    """
    Invert configured prewhitening in the Fourier domain channel by channel.

    Frequencies where the whitening response is zero are set to zero, including
    DC for first differences, instead of emitting infinities.
    """
    prewhitening_type = _prewhitening_type(stft_config)
    if not prewhitening_type:
        logger.debug("No prewhitening was defined -- skipping recoloring")
        return stft_obj

    stft_obj = stft_obj.copy(deep=True)
    frequencies = np.asarray(stft_obj.frequency.data)
    for channel_id in stft_obj.data_vars:
        response = _whitening_response(
            prewhitening_type,
            frequencies,
            sample_rate,
            results.get(channel_id),
        )
        valid = ~np.isclose(response, 0.0)
        data = np.asarray(stft_obj[channel_id].data)
        recolored = np.zeros_like(data)
        np.divide(data, response, out=recolored, where=valid)
        stft_obj[channel_id].data = recolored
        if channel_id in results:
            stft_obj[channel_id].attrs["prewhitening"] = results[channel_id].as_dict()
    return stft_obj
