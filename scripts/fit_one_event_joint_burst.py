#!/usr/bin/env python3
"""Run one geometry-constrained joint fit from strict observation products."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import dynesty
import numpy as np
from one_event_workflow import arrays_sha256, load_config, sample_time_axis_ns

import radio_pipeline
from radio_pipeline.fitting import (
    AssociationHypothesis,
    ComponentMatch,
    ComponentWindow,
    FitSettings,
    GeometryConstraint,
    JointFitRequest,
    fit_joint_event,
    load_band_observation_product,
)
from radio_pipeline.fitting.products import sha256_file


def _require_accepted_status(status: str) -> None:
    if status != "provisional_pending_owner_approval":
        raise RuntimeError(f"joint fit failed acceptance gates: {status}")


def _arrays_sha256(*arrays: np.ndarray) -> str:
    return arrays_sha256(*arrays)


def _require_locked_array_identity(observation, resolution: dict) -> None:
    instrument = observation.instrument
    if (
        _arrays_sha256(
            observation.frequency_mhz,
            observation.channel_width_mhz,
        )
        != resolution[f"{instrument}_frequency_grid_sha256"]
    ):
        raise ValueError(f"{instrument} authoritative frequency grid changed")
    if _arrays_sha256(observation.valid) != resolution[f"{instrument}_valid_mask_sha256"]:
        raise ValueError(f"{instrument} accepted valid support changed")


def _require_locked_product_metadata(
    product,
    instrument: str,
    resolution: dict,
) -> None:
    expected_time0 = int(resolution[f"{instrument}_time0_unix_ns"])
    if int(product["time0_unix_ns"]) != expected_time0:
        raise ValueError(f"{instrument} locked crop origin changed")
    if "noise_estimation_mask" not in product:
        raise ValueError(f"{instrument} product lacks off-pulse mask")
    if (
        _arrays_sha256(product["noise_estimation_mask"])
        != resolution[f"{instrument}_off_pulse_mask_sha256"]
    ):
        raise ValueError(f"{instrument} locked off-pulse support changed")
    if (
        _arrays_sha256(product["waterfall"])
        != resolution[f"{instrument}_waterfall_sha256"]
    ):
        raise ValueError(f"{instrument} locked waterfall pixels changed")
    if (
        _arrays_sha256(product["noise_std"])
        != resolution[f"{instrument}_noise_std_sha256"]
    ):
        raise ValueError(f"{instrument} locked noise estimates changed")
    time_axis_ns = sample_time_axis_ns(
        time0_unix_ns=int(product["time0_unix_ns"]),
        sample_interval_s=float(product["sample_interval_s"]),
        sample_count=int(product["waterfall"].shape[1]),
    )
    if (
        _arrays_sha256(time_axis_ns)
        != resolution[f"{instrument}_time_axis_sha256"]
    ):
        raise ValueError(f"{instrument} locked time axis changed")


def _runtime_preflight(repo_root: Path) -> dict[str, str]:
    expected_package = (repo_root / "radio_pipeline").resolve()
    actual_package = Path(radio_pipeline.__file__).resolve().parent
    if actual_package != expected_package:
        raise RuntimeError("radio_pipeline was not imported from this checkout")
    dynesty_path = Path(dynesty.__file__).resolve()
    if not str(dynesty.__version__).startswith("3.1."):
        raise RuntimeError("locked workflow requires dynesty 3.1.x")
    if any("dsa110-FLITS" in entry for entry in sys.path):
        raise RuntimeError("retired editable FLITS runtime detected")
    return {
        "radio_pipeline": str(actual_package),
        "dynesty": str(dynesty_path),
        "dynesty_version": str(dynesty.__version__),
    }


def _dsa_product_dm_bounds(
    config: dict,
    product_dm_pc_cm3: float,
) -> tuple[float, float] | None:
    """Propagate the reviewed raw-input interval through the applied correction."""

    dsa = config["dsa"]
    if "reference_minus_raw_dm_interval_pc_cm3" not in dsa:
        return None
    residual_low, residual_high = map(
        float,
        dsa["reference_minus_raw_dm_interval_pc_cm3"],
    )
    residual_nominal = float(dsa["reference_minus_raw_dm_pc_cm3"])
    accepted_dm = float(dsa["accepted_reference_dm_pc_cm3"])
    if not residual_low < residual_nominal < residual_high:
        raise ValueError("DSA nominal input-DM residual must lie inside its reviewed interval")
    method = dsa["input_dm_method"]
    if method == "inferred_raw_reference_row_timing":
        nominal_input_dm = float(dsa["input_dm_pc_cm3"])
        if not np.isclose(
            nominal_input_dm,
            accepted_dm - residual_nominal,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("DSA input-DM point and residual interval use inconsistent coordinates")
        raw_dm_bounds = accepted_dm - residual_high, accepted_dm - residual_low
        return (
            float(product_dm_pc_cm3) + raw_dm_bounds[0] - nominal_input_dm,
            float(product_dm_pc_cm3) + raw_dm_bounds[1] - nominal_input_dm,
        )
    if method == "accepted_product_dm_nominal_with_residual_bound":
        if not np.isclose(
            float(dsa["input_dm_pc_cm3"]),
            accepted_dm,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("DSA accepted-product nominal must equal the accepted reference DM")
        return (
            float(product_dm_pc_cm3) - residual_high,
            float(product_dm_pc_cm3) - residual_low,
        )
    raise ValueError(f"unsupported DSA input-DM method: {method}")


def _request(
    config: dict,
    chime_path: Path,
    dsa_path: Path,
    geometry_path: Path,
) -> JointFitRequest:
    settings = config["joint_fit"]
    observations = (
        load_band_observation_product(chime_path),
        load_band_observation_product(dsa_path),
    )
    dsa_config = config["dsa"]
    expected_dsa_product_dm = float(config["chime"]["anchor_dm_pc_cm3"])
    product_dm_bounds = _dsa_product_dm_bounds(
        config,
        observations[1].dispersion.product_dm_pc_cm3,
    )
    if not np.isclose(
        observations[1].dispersion.product_dm_pc_cm3,
        expected_dsa_product_dm,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("DSA fit product DM differs from the coherent anchor")
    if product_dm_bounds is not None:
        observations = (
            observations[0],
            replace(
                observations[1],
                dispersion=replace(
                    observations[1].dispersion,
                    product_dm_bounds_pc_cm3=product_dm_bounds,
                    product_dm_bound_source=dsa_config["input_dm_bound_source"],
                ),
            ),
        )
    expected_inputs = config["input_sha256"]
    expected_chime_hashes = {
        "raw_chime_h5": expected_inputs["raw_chime_h5"],
        "accepted_chime_support": config["chime"]["accepted_support"]["mask_sha256"],
    }
    expected_dsa_hashes = {
        "raw_dsa_filterbank": expected_inputs["raw_dsa_filterbank"],
        "accepted_dsa_support": config["dsa"]["accepted_support"]["mask_sha256"],
    }
    if config["workflow"].get("observation_source") != "raw_instrument_products_only":
        expected_chime_hashes = {
            "raw_chime_h5": expected_inputs["raw_chime_h5"],
            "accepted_chime_reference": expected_inputs["accepted_chime_reference"],
        }
        expected_dsa_hashes = {
            "raw_dsa_filterbank": expected_inputs["raw_dsa_filterbank"],
            "accepted_dsa_reference": expected_inputs["accepted_dsa_reference"],
        }
    if observations[0].input_sha256 != expected_chime_hashes:
        raise ValueError("CHIME observation input hashes differ from configuration")
    if observations[1].input_sha256 != expected_dsa_hashes:
        raise ValueError("DSA observation input hashes differ from configuration")
    resolution = settings["resolution"]
    for observation, product_path in zip(observations, (chime_path, dsa_path), strict=True):
        expected_shape = tuple(map(int, resolution[f"{observation.instrument}_shape"]))
        expected_sample = float(resolution[f"{observation.instrument}_sample_interval_s"])
        if observation.waterfall.shape != expected_shape:
            raise ValueError(f"{observation.instrument} locked crop shape changed")
        if not np.isclose(
            observation.sample_interval_s,
            expected_sample,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(f"{observation.instrument} locked time resolution changed")
        with np.load(product_path, allow_pickle=False) as product:
            _require_locked_product_metadata(
                product,
                observation.instrument,
                resolution,
            )
            for axis in ("frequency", "time"):
                key = f"{axis}_bin_factor"
                if key not in product:
                    raise ValueError(f"{observation.instrument} product lacks {key}")
                expected = int(resolution[f"{observation.instrument}_{axis}_bin_factor"])
                if int(product[key]) != expected:
                    raise ValueError(f"{observation.instrument} locked {axis} bin factor changed")
        _require_locked_array_identity(observation, resolution)
    geometry_data = json.loads(geometry_path.read_text())
    if geometry_data["event_binding_sha256"] != config["event_binding_sha256"]:
        raise ValueError("geometry constraint belongs to another configuration")
    geometry = GeometryConstraint(
        epoch_unix_ns=int(geometry_data["epoch_unix_ns"]),
        source_icrs=geometry_data["source_icrs"],
        site_delay_s=geometry_data["site_delay_s"],
        site_delay_sigma_s=geometry_data["site_delay_sigma_s"],
        clock_sigma_s=geometry_data["clock_sigma_s"],
        projection_disagreement_s=float(geometry_data["projection_disagreement_s"]),
        reference_frequency_mhz=float(geometry_data["reference_frequency_mhz"]),
    )
    components = tuple(
        ComponentWindow(
            instrument=row["instrument"],
            component_id=row["component_id"],
            center_sample=float(row["center_sample"]),
            half_width_samples=float(row["half_width_samples"]),
            width_bounds_s=tuple(map(float, row["width_bounds_s"])),
            width_index_bounds=tuple(map(float, row.get("width_index_bounds", [-2.0, 2.0]))),
        )
        for row in settings["components"]
    )
    associations = tuple(
        AssociationHypothesis(
            name=row["name"],
            matches=tuple(
                ComponentMatch(
                    latent_id=match["latent_id"],
                    chime_component_id=match["chime_component_id"],
                    dsa_component_id=match["dsa_component_id"],
                )
                for match in row["matches"]
            ),
        )
        for row in settings["associations"]
    )
    sampler = settings["sampler"]
    fit_settings = FitSettings(
        dm_bounds_pc_cm3=tuple(map(float, settings["dm_bounds_pc_cm3"])),
        morphologies=tuple(settings["morphologies"]),
        scattering_tau_1ghz_bounds_s=tuple(map(float, settings["scattering_tau_1ghz_bounds_s"])),
        scattering_alpha_bounds=tuple(map(float, settings["scattering_alpha_bounds"])),
        gain_variance=float(settings["gain_variance"]),
        seed=int(sampler["seed"]),
        nlive=int(sampler["nlive"]),
        dlogz=float(sampler["dlogz"]),
        sample=sampler.get("sample", "rwalk"),
        pool_size=int(sampler.get("pool_size", 1)),
        checkpoint_dir=str(Path(config["paths"]["output_root"]) / ".checkpoints"),
        resume=bool(sampler.get("resume", True)),
        maximum_projection_disagreement_s=float(
            settings.get("maximum_projection_disagreement_s", 5.0e-7)
        ),
        maximum_reduced_residual_power=float(
            settings["acceptance"]["maximum_reduced_residual_power"]
        ),
        maximum_structured_residual_correlation=float(
            settings["acceptance"]["maximum_structured_residual_correlation"]
        ),
        posterior_edge_fraction=float(settings["acceptance"]["posterior_edge_fraction"]),
        maximum_prior_edge_mass=float(settings["acceptance"]["maximum_prior_edge_mass"]),
        minimum_supported_run_weight=float(
            settings["acceptance"].get("minimum_supported_run_weight", 0.01)
        ),
        maximum_timing_offset_sigma=float(
            settings["acceptance"]["maximum_timing_offset_sigma"]
        ),
        maximum_timing_offset_tail_mass=float(
            settings["acceptance"]["maximum_timing_offset_tail_mass"]
        ),
    )
    return JointFitRequest(
        observations=observations,
        geometry=geometry,
        components=components,
        associations=associations,
        settings=fit_settings,
    )


def run(
    config: dict,
    *,
    chime_path: Path,
    dsa_path: Path,
    geometry_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    runtime = _runtime_preflight(repo_root)
    request = _request(config, chime_path, dsa_path, geometry_path)
    result = fit_joint_event(request)
    shutil.rmtree(output_dir / ".checkpoints", ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_result_path = output_dir / "fit-result.json"
    posterior_path = output_dir / "posterior.npz"
    model_path = output_dir / "model-products.npz"
    np.savez_compressed(
        posterior_path,
        run_weights=result.run_weights,
        **{
            f"run_{index}_{field}": value
            for index, run in enumerate(result.runs)
            for field, value in (
                ("samples", run.samples),
                ("sample_weights", run.sample_weights),
                ("parameter_names", np.asarray(run.parameter_names)),
                ("morphology", np.asarray(run.morphology)),
                ("association", np.asarray(run.association)),
                ("log_evidence", np.asarray(run.log_evidence)),
                ("log_evidence_error", np.asarray(run.log_evidence_error)),
            )
        },
    )
    np.savez_compressed(model_path, **result.model_products)
    fit_payload = {
        "schema_version": 1,
        "status": result.status,
        "event": config["event"],
        "event_binding_sha256": config["event_binding_sha256"],
        "reference_frequency_mhz": 400.0,
        "shared_absolute_dm_pc_cm3": result.dm_pc_cm3,
        "geocentric_unscattered_toa_unix_ns": (result.geocentric_toa_unix_ns),
        "topocentric_toa_unix_ns": result.topocentric_toa_unix_ns,
        "diagnostics": result.diagnostics,
        "runs": [
            {
                "morphology": run.morphology,
                "association": run.association,
                "log_evidence": run.log_evidence,
                "log_evidence_error": run.log_evidence_error,
                "diagnostics": run.diagnostics,
            }
            for run in result.runs
        ],
        "compact_products": {
            "posterior.npz": sha256_file(posterior_path),
            "model-products.npz": sha256_file(model_path),
        },
        "fit_inputs": {
            "geometry_constraint": sha256_file(geometry_path),
            "chime_observation": sha256_file(chime_path),
            "dsa_observation": sha256_file(dsa_path),
        },
    }
    fit_result_path.write_text(json.dumps(fit_payload, indent=2, allow_nan=False) + "\n")
    provenance = {
        "schema_version": 1,
        "status": result.status,
        "event": config["event"],
        "event_binding_sha256": config["event_binding_sha256"],
        "inputs": {
            "config": sha256_file(config["_config_path"]),
            "chime_observation": sha256_file(chime_path),
            "dsa_observation": sha256_file(dsa_path),
            "geometry_constraint": sha256_file(geometry_path),
        },
        "code": {
            "joint_burst": sha256_file(repo_root / "radio_pipeline/fitting/joint_burst.py"),
            "products": sha256_file(repo_root / "radio_pipeline/fitting/products.py"),
            "pulse_kernels": sha256_file(repo_root / "radio_pipeline/fitting/_pulse_kernels.py"),
            "runner": sha256_file(Path(__file__)),
        },
        "runtime": runtime,
        "outputs": {
            "fit-result.json": sha256_file(fit_result_path),
            "posterior.npz": sha256_file(posterior_path),
            "model-products.npz": sha256_file(model_path),
        },
    }
    provenance_path = output_dir / "run-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    _require_accepted_status(result.status)
    return fit_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chime-observation", type=Path, required=True)
    parser.add_argument("--dsa-observation", type=Path, required=True)
    parser.add_argument("--geometry-constraint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(args.config.resolve())
    result = run(
        config,
        chime_path=args.chime_observation,
        dsa_path=args.dsa_observation,
        geometry_path=args.geometry_constraint,
        output_dir=args.output_dir,
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps({"status": result["status"], "event": result["event"]}))


if __name__ == "__main__":
    main()
