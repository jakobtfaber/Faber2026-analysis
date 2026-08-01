#!/usr/bin/env python3
"""Replay raw timing authorities without constructing fit observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from one_event_workflow import load_config

K_DM_S_MHZ2 = 4148.808
REFERENCE_FREQUENCY_MHZ = 400.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_trigger(entry: dict) -> dict:
    specnum = int(entry["specnum"])
    itime = specnum // 4 + 1907
    elapsed_true_s = itime * 262.144e-6
    dt_f32 = np.float32(np.float32(262.144) * np.float32(1.0e-6))
    serialized_token_s = float(format(float(np.float32(np.float32(itime) * dt_f32)), ".6g"))
    correction_s = elapsed_true_s - serialized_token_s
    recovered_mjd = float(entry["mjds_T2"]) + correction_s / 86400.0
    return {
        "specnum": specnum,
        "itime": itime,
        "elapsed_true_s": elapsed_true_s,
        "serialized_token_s": serialized_token_s,
        "correction_us": correction_s * 1.0e6,
        "recovered_mjd": recovered_mjd,
    }


def audit_chime(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        time0 = handle["time0"][:]
        frequency_mhz = np.asarray(handle["index_map/freq"][:]["centre"], dtype=float)
        seconds = time0["ctime"].astype(np.longdouble) + time0["ctime_offset"].astype(
            np.longdouble
        )
        fpga_count = time0["fpga_count"].astype(np.longdouble)
        delta_time_s = np.longdouble(str(handle.attrs["delta_time"]))
        affine_anchor_s = seconds[0] - fpga_count[0] * delta_time_s
        affine_residual_s = seconds - (affine_anchor_s + fpga_count * delta_time_s)

        x = 1.0 / frequency_mhz**2 - 1.0 / REFERENCE_FREQUENCY_MHZ**2
        design = np.column_stack((np.ones_like(x), K_DM_S_MHZ2 * x))
        centered = np.asarray(seconds - seconds.mean(), dtype=float)
        intercept_centered, capture_dm = np.linalg.lstsq(design, centered, rcond=None)[0]
        window_start_400_s = float(intercept_centered + seconds.mean())
        schedule_residual_s = np.asarray(
            seconds - (window_start_400_s + K_DM_S_MHZ2 * capture_dm * x), dtype=float
        )

        event_date = str(handle.attrs["event_date"]).replace("T ", "T")
        event_unix_s = datetime.fromisoformat(event_date).replace(tzinfo=UTC).timestamp()
        ntime = int(handle["tiedbeam_baseband"].shape[-1])
        window_stop_400_s = window_start_400_s + ntime * float(delta_time_s)
        event_inside_window = window_start_400_s <= event_unix_s <= window_stop_400_s
        if not event_inside_window:
            raise RuntimeError("CHIME event timestamp lies outside the 400 MHz capture window")
        maximum_affine_residual_ns = float(np.max(np.abs(affine_residual_s)) * 1.0e9)
        if maximum_affine_residual_ns > 1.0:
            raise RuntimeError("CHIME ctime fields disagree with the FPGA counter")
        maximum_schedule_residual_us = float(np.max(np.abs(schedule_residual_s)) * 1.0e6)
        if maximum_schedule_residual_us > 5.0:
            raise RuntimeError("CHIME channel start times do not follow a cold-plasma schedule")
        return {
            "status": "pass_internal_counter_and_capture_schedule_cross_check",
            "event_id": int(handle.attrs["event_id"]),
            "event_date_utc": event_date,
            "event_date_unix_s": event_unix_s,
            "embedded_baseband_analysis_git_sha": str(
                handle.attrs["baseband-analysis_git_sha"]
            ),
            "archive_version": str(handle.attrs["archive_version"]),
            "channel_count": int(frequency_mhz.size),
            "delta_time_s": float(delta_time_s),
            "maximum_fpga_affine_residual_ns": maximum_affine_residual_ns,
            "capture_schedule_dm_pc_cm3": float(capture_dm),
            "capture_schedule_dm_role": "window-placement diagnostic, not a fitted burst DM",
            "capture_schedule_maximum_residual_us": maximum_schedule_residual_us,
            "capture_window_start_unix_400_s": window_start_400_s,
            "capture_window_stop_unix_400_s": window_stop_400_s,
            "event_date_inside_capture_window": event_inside_window,
            "external_clock_certification": "not supplied",
        }


def audit_dsa(path: Path, trigger_entry: dict) -> dict:
    from blimpy import Waterfall

    replay = recover_trigger(trigger_entry)
    recorded_mjd = float(trigger_entry["mjd_trigger_exact"])
    replay_difference_ns = (replay["recovered_mjd"] - recorded_mjd) * 86400.0 * 1.0e9
    if abs(replay_difference_ns) > 1000.0:
        raise RuntimeError("DSA trigger recovery does not reproduce the locked value")
    if trigger_entry.get("status") != "VERIFIED":
        raise RuntimeError("DSA trigger authority is not verified")

    reader = Waterfall(str(path), load_data=False)
    header_tstart_mjd = float(reader.header["tstart"])
    sample_time_s = float(reader.header["tsamp"])
    header_to_trigger_s = (recorded_mjd - header_tstart_mjd) * 86400.0
    header_to_trigger_samples = header_to_trigger_s / sample_time_s
    return {
        "trigger_recovery_status": "pass_tolerance_bounded_replay",
        "trigger_mjd_utc": recorded_mjd,
        "replayed_trigger_mjd_utc": replay["recovered_mjd"],
        "replay_difference_ns": replay_difference_ns,
        "trigger_replay": replay,
        "filterbank_tstart_mjd": header_tstart_mjd,
        "filterbank_sample_interval_s": sample_time_s,
        "rounded_header_to_trigger_s": header_to_trigger_s,
        "rounded_header_to_trigger_samples": header_to_trigger_samples,
        "filterbank_sample_zero_status": "blocked_missing_exact_trigger_to_sample_mapping",
        "filterbank_tstart_use": "diagnostic_only_not_an_absolute_time_authority",
        "fit_observation_time_origin_eligible": False,
    }


def build_receipt(config_path: Path) -> dict:
    config = load_config(config_path)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    for key in ("raw_chime_h5", "raw_dsa_filterbank", "trigger_recovery"):
        actual = sha256_file(paths[key])
        if actual != config["input_sha256"][key]:
            raise RuntimeError(f"{key} hash differs from configuration")
    triggers = json.loads(paths["trigger_recovery"].read_text())
    event = config["event"]
    if event not in triggers:
        raise RuntimeError(f"trigger recovery lacks event {event}")
    dsa = audit_dsa(paths["raw_dsa_filterbank"], triggers[event])
    if dsa.get("fit_observation_time_origin_eligible") is not False:
        raise RuntimeError("DSA timing must remain ineligible without sample-zero authority")
    if dsa.get("filterbank_sample_zero_status") != (
        "blocked_missing_exact_trigger_to_sample_mapping"
    ):
        raise RuntimeError("DSA sample-zero provenance hold is absent")
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "status": "timing_replayed_fit_input_blocked",
        "event": event,
        "event_binding_sha256": config["event_binding_sha256"],
        "reference_frequency_mhz": REFERENCE_FREQUENCY_MHZ,
        "chime": audit_chime(paths["raw_chime_h5"]),
        "dsa": dsa,
        "provenance": {
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "raw_chime_h5": str(paths["raw_chime_h5"]),
            "raw_chime_h5_sha256": config["input_sha256"]["raw_chime_h5"],
            "raw_dsa_filterbank": str(paths["raw_dsa_filterbank"]),
            "raw_dsa_filterbank_sha256": config["input_sha256"]["raw_dsa_filterbank"],
            "trigger_recovery": str(paths["trigger_recovery"]),
            "trigger_recovery_sha256": config["input_sha256"]["trigger_recovery"],
            "script_sha256": sha256_file(Path(__file__)),
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
        },
    }


def publish_receipt(config_path: Path, output_path: Path) -> dict:
    receipt = build_receipt(config_path)
    payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish_receipt(args.config, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
