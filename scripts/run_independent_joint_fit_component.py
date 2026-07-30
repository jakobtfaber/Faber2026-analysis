#!/usr/bin/env python3
"""Evaluate one configured morphology into a shared resumable checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from fit_one_event_joint_burst import _request
from one_event_workflow import load_config

from radio_pipeline.fitting.joint_burst import _fit_one
from radio_pipeline.fitting.products import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chime-observation", type=Path, required=True)
    parser.add_argument("--dsa-observation", type=Path, required=True)
    parser.add_argument("--geometry-constraint", type=Path, required=True)
    parser.add_argument("--morphology", choices=("gaussian", "scattering"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    request = _request(
        config,
        args.chime_observation,
        args.dsa_observation,
        args.geometry_constraint,
    )
    if args.morphology not in request.settings.morphologies:
        raise ValueError("requested morphology is absent from the frozen configuration")
    request = replace(
        request,
        settings=replace(
            request.settings,
            checkpoint_dir=str(args.checkpoint_dir),
            resume=True,
        ),
    )
    if len(request.associations) != 1:
        raise ValueError("independent runner requires one configured association")
    fit = _fit_one(request, request.associations[0], args.morphology)
    checkpoints = sorted(args.checkpoint_dir.glob(f"{args.morphology}-*.save"))
    if len(checkpoints) != 1:
        raise RuntimeError("expected exactly one completed morphology checkpoint")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "event": config["event"],
        "event_binding_sha256": config["event_binding_sha256"],
        "morphology": fit.morphology,
        "association": fit.association,
        "log_evidence": fit.log_evidence,
        "log_evidence_error": fit.log_evidence_error,
        "sample_count": int(fit.samples.shape[0]),
        "finite_samples": bool(np.isfinite(fit.samples).all()),
        "checkpoint": {
            "path": str(checkpoints[0]),
            "sha256": sha256_file(checkpoints[0]),
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.config,
                args.chime_observation,
                args.dsa_observation,
                args.geometry_constraint,
            )
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
