from __future__ import annotations

import pickle

import numpy as np
import pytest

from radio_pipeline.fitting.joint_burst import (
    BandObservation,
    DispersionState,
    _gain_marginal_band,
    _gain_marginal_band_reference,
)


def _observation(
    rng: np.random.Generator,
    *,
    nfrequency: int = 7,
    ntime: int = 31,
) -> BandObservation:
    valid = rng.random((nfrequency, ntime)) > 0.23
    valid[0] = False
    valid[1, :] = False
    valid[1, 4] = True
    waterfall = rng.normal(size=(nfrequency, ntime))
    noise = np.exp(rng.uniform(np.log(1.0e-3), np.log(1.0e2), waterfall.shape))
    waterfall[~valid] = np.nan
    noise[~valid] = np.nan
    return BandObservation(
        instrument="chime",
        waterfall=waterfall,
        valid=valid,
        frequency_mhz=np.linspace(400.0, 800.0, nfrequency),
        channel_width_mhz=np.full(nfrequency, 0.1),
        noise_std=noise,
        sample_interval_s=1.0e-4,
        time0_unix_ns=1_700_000_000_000_000_000,
        reference_frequency_mhz=400.0,
        dispersion=DispersionState(0.0, 100.0, 0.0, 100.0, "coherent_anchor"),
    )


def test_scalar_gain_integral_matches_matrix_oracle_at_10000_points() -> None:
    rng = np.random.default_rng(20260730)
    observation = _observation(rng)
    for index in range(10_000):
        kernel = rng.normal(size=(1, *observation.waterfall.shape))
        if index % 5 == 0:
            kernel *= 1.0e-8
        elif index % 5 == 1:
            kernel *= 1.0e4
        gain_variance = 10.0 ** rng.uniform(-8.0, 8.0)
        fast, fast_model = _gain_marginal_band(
            observation,
            kernel,
            gain_variance,
            return_model=index % 1000 == 0,
        )
        oracle, oracle_model = _gain_marginal_band_reference(
            observation,
            kernel,
            gain_variance,
            return_model=index % 1000 == 0,
        )
        assert abs(fast - oracle) <= 1.0e-6
        if fast_model is not None:
            assert oracle_model is not None
            np.testing.assert_allclose(
                fast_model[observation.valid],
                oracle_model[observation.valid],
                rtol=2.0e-13,
                atol=1.0e-12,
            )


def test_multiple_components_still_use_general_matrix_integral() -> None:
    rng = np.random.default_rng(17)
    observation = _observation(rng)
    kernels = rng.normal(size=(3, *observation.waterfall.shape))
    actual, model = _gain_marginal_band(
        observation,
        kernels,
        3.0,
        return_model=True,
    )
    expected, expected_model = _gain_marginal_band_reference(
        observation,
        kernels,
        3.0,
        return_model=True,
    )
    assert actual == expected
    np.testing.assert_array_equal(model, expected_model)


def test_observation_owns_immutable_inputs_and_cached_statistics() -> None:
    rng = np.random.default_rng(29)
    source = rng.normal(size=(4, 19))
    observation = BandObservation(
        instrument="dsa",
        waterfall=source,
        valid=np.ones_like(source, dtype=bool),
        frequency_mhz=np.linspace(1300.0, 1500.0, 4),
        channel_width_mhz=np.full(4, 0.1),
        noise_std=np.ones_like(source),
        sample_interval_s=1.0e-4,
        time0_unix_ns=1_700_000_000_000_000_000,
        reference_frequency_mhz=400.0,
        dispersion=DispersionState(0.0, 100.0, 0.0, 100.0, "raw_filterbank"),
        input_sha256={"raw": "a" * 64},
    )
    kernel = rng.normal(size=(1, *source.shape))
    before = _gain_marginal_band(observation, kernel, 2.0, return_model=False)[0]
    source[:] = 99.0
    after = _gain_marginal_band(observation, kernel, 2.0, return_model=False)[0]
    assert before == after
    assert not observation.waterfall.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        observation.waterfall.flags.writeable = True
    with pytest.raises(TypeError):
        observation.input_sha256["raw"] = "b" * 64
    with pytest.raises(TypeError):
        observation.input_sha256 |= {"raw": "b" * 64}
    with pytest.raises(AttributeError):
        observation.waterfall = np.zeros_like(observation.waterfall)
    restored = pickle.loads(pickle.dumps(observation))
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        restored.waterfall.flags.writeable = True
    with pytest.raises(TypeError):
        restored.input_sha256["raw"] = "b" * 64
