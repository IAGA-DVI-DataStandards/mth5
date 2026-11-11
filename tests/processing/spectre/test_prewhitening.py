from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from mth5.processing.spectre.prewhitening import (
    apply_prewhitening,
    apply_recoloring,
)
from mth5.processing.spectre.stft import run_ts_to_stft_scipy
from mth5.mth5 import MTH5
from mt_metadata.processing.fourier_coefficients import Decimation


def _config(prewhitening_type, ar_order=0, ma_order=0):
    return SimpleNamespace(
        prewhitening_type=prewhitening_type,
        prewhitening_ar_order=ar_order,
        prewhitening_ma_order=ma_order,
    )


def _time_series_dataset(values):
    return xr.Dataset(
        {"hx": ("time", values)},
        coords={"time": np.arange(len(values), dtype="timedelta64[s]")},
    )


def _fourier_dataset(values, frequencies):
    return xr.Dataset(
        {"hx": (("time", "frequency"), values[np.newaxis, :])},
        coords={
            "time": np.array([0], dtype="timedelta64[s]"),
            "frequency": frequencies,
        },
    )


def test_first_difference_uses_discrete_difference():
    dataset = _time_series_dataset(np.array([2.0, 5.0, 9.0, 14.0]))

    whitened, results = apply_prewhitening(_config("first difference"), dataset)

    np.testing.assert_array_equal(whitened.hx.data, np.array([3.0, 4.0, 5.0]))
    assert whitened.time.data[0] == np.timedelta64(1, "s")
    assert results == {}


def test_first_difference_recoloring_uses_exact_response():
    frequencies = np.array([0.0, 0.125, 0.25, 0.375])
    original = np.array([3.0 + 1.0j, 2.0 - 4.0j, -1.0 + 2.0j, 5.0 - 3.0j])
    response = 1.0 - np.exp(-2.0j * np.pi * frequencies)
    whitened = _fourier_dataset(original * response, frequencies)

    recolored = apply_recoloring(
        _config("first difference"),
        whitened,
        sample_rate=1.0,
        results={},
    )

    assert recolored.hx.data[0, 0] == 0.0
    np.testing.assert_allclose(recolored.hx.data[0, 1:], original[1:])


def test_ar_prewhitening_returns_innovations_and_recolors_per_channel():
    rng = np.random.default_rng(42)
    innovations = rng.normal(size=512)
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for index in range(1, len(samples)):
        samples[index] = 0.7 * samples[index - 1] + innovations[index]

    whitened, results = apply_prewhitening(
        _config("ARMA", ar_order=1),
        _time_series_dataset(samples),
    )

    assert abs(np.corrcoef(whitened.hx.data[:-1], whitened.hx.data[1:])[0, 1]) < abs(
        np.corrcoef(samples[:-1], samples[1:])[0, 1]
    )
    assert results["hx"].ar_coefficients[0] == pytest.approx(0.7, abs=0.1)
    assert results["hx"].ma_coefficients == ()

    frequencies = np.array([0.1, 0.2, 0.3])
    original = np.array([2.0 + 3.0j, -1.0 + 4.0j, 5.0 - 2.0j])
    coefficient = results["hx"].ar_coefficients[0]
    response = 1.0 - coefficient * np.exp(-2.0j * np.pi * frequencies)
    recolored = apply_recoloring(
        _config("ARMA", ar_order=1),
        _fourier_dataset(original * response, frequencies),
        sample_rate=1.0,
        results=results,
    )

    np.testing.assert_allclose(recolored.hx.data[0], original)
    assert recolored.hx.attrs["prewhitening"]["ar_coefficients"] == pytest.approx(
        list(results["hx"].ar_coefficients)
    )


def test_nonzero_ma_order_is_not_silently_accepted():
    with pytest.raises(NotImplementedError, match="prewhitening_ma_order"):
        apply_prewhitening(
            _config("ARMA", ar_order=1, ma_order=1),
            _time_series_dataset(np.arange(16, dtype=float)),
        )


def test_ar_prewhitening_rejects_missing_samples():
    values = np.arange(16, dtype=float)
    values[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        apply_prewhitening(
            _config("ARMA", ar_order=1),
            _time_series_dataset(values),
        )


def test_stft_carries_ar_coefficients_to_fourier_data():
    rng = np.random.default_rng(42)
    innovations = rng.normal(size=256)
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for index in range(1, len(samples)):
        samples[index] = 0.6 * samples[index - 1] + innovations[index]

    decimation = Decimation()
    decimation.decimation.sample_rate = 1.0
    decimation.stft.window.num_samples = 32
    decimation.stft.window.overlap = 16
    decimation.stft.prewhitening_type = "ARMA"
    decimation.stft.prewhitening_ar_order = 1
    decimation.stft.recoloring = True

    spectrogram = run_ts_to_stft_scipy(
        decimation,
        _time_series_dataset(samples),
    )

    assert "prewhitening" in spectrogram.dataset.hx.attrs
    assert len(spectrogram.dataset.hx.attrs["prewhitening"]["ar_coefficients"]) == 1
    assert np.isfinite(spectrogram.dataset.hx.data).all()


def test_fc_archive_preserves_prewhitening_result(tmp_path):
    frequencies = np.array([0.1, 0.2])
    dataset = xr.Dataset(
        {
            "hx": (
                ("time", "frequency"),
                np.array([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]]),
                {
                    "prewhitening": {
                        "ar_coefficients": [0.8],
                        "ma_coefficients": [],
                    },
                    "prewhitening_type": "ARMA",
                    "prewhitening_fit_domain": "full run",
                    "prewhitening_response_convention": (
                        "A(z) = 1 - sum(phi_k z^-k); E(f) = A(f) X(f)"
                    ),
                },
            )
        },
        coords={
            "time": np.array(
                ["2020-01-01T00:00:00", "2020-01-01T00:00:01"],
                dtype="datetime64[s]",
            ),
            "frequency": frequencies,
        },
    )
    filename = tmp_path / "prewhitening.h5"

    with MTH5(file_version="0.1.0").open_mth5(filename, mode="w") as mth5:
        station = mth5.add_station("test")
        fc_group = station.fourier_coefficients_group.add_fc_group("test_run")
        decimation = fc_group.add_decimation_level("0")
        decimation.from_xarray(dataset, sample_rate_decimation_level=1.0)
        metadata = decimation.get_channel("hx").metadata

    assert metadata.prewhitening_type == "ARMA"
    assert metadata.prewhitening_fit_domain == "full run"
    assert metadata.prewhitening_ar_coefficients == [0.8]
    assert metadata.prewhitening_ma_coefficients == []
