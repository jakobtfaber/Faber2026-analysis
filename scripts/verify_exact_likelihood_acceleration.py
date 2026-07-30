#!/usr/bin/env python3
"""Verify and benchmark the exact one-component likelihood reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from fit_one_event_joint_burst import _request

from radio_pipeline.fitting.joint_burst import (
    _component_kernels,
    _gain_marginal_band,
    _gain_marginal_band_reference,
    _layout,
    _prior_transform,
    _values,
)


def _unit_points(count: int, ndim: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Cover the prior interior, a central fit region, and every prior edge."""

    rng = np.random.default_rng(seed)
    points = np.empty((count, ndim), dtype=float)
    labels: list[str] = []
    for index in range(count):
        category = index % 10
        if category < 7:
            points[index] = rng.uniform(1.0e-9, 1.0 - 1.0e-9, ndim)
            labels.append("prior")
        elif category < 9:
            points[index] = np.clip(rng.normal(0.5, 0.06, ndim), 1.0e-9, 1.0 - 1.0e-9)
            labels.append("central")
        else:
            points[index] = rng.uniform(1.0e-9, 1.0 - 1.0e-9, ndim)
            coordinate = (index // 10) % ndim
            points[index, coordinate] = (
                1.0e-9 if (index // (10 * ndim)) % 2 == 0 else 1.0 - 1.0e-9
            )
            labels.append("edge")
    return points, labels


def run(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text())
    request = _request(
        config,
        args.chime_observation,
        args.dsa_observation,
        args.geometry_constraint,
    )
    hypothesis = request.associations[0]
    layout = _layout(request, hypothesis, args.morphology)
    points, labels = _unit_points(args.points, len(layout.parameters), args.seed)
    unit_sha256 = hashlib.sha256(np.ascontiguousarray(points).view(np.uint8)).hexdigest()

    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    failures: list[dict[str, object]] = []
    fast_seconds = 0.0
    reference_seconds = 0.0
    kernel_seconds = 0.0
    for index, unit in enumerate(points):
        theta = _prior_transform(unit, layout)
        values = _values(theta, layout)
        started = time.perf_counter()
        kernels = [
            _component_kernels(request, observation, layout, values, args.morphology)
            for observation in request.observations
        ]
        kernel_seconds += time.perf_counter() - started

        started = time.perf_counter()
        fast = sum(
            _gain_marginal_band(
                observation,
                kernel,
                request.settings.gain_variance,
            )[0]
            for observation, kernel in zip(request.observations, kernels, strict=True)
        )
        fast_seconds += time.perf_counter() - started

        started = time.perf_counter()
        reference = sum(
            _gain_marginal_band_reference(
                observation,
                kernel,
                request.settings.gain_variance,
            )[0]
            for observation, kernel in zip(request.observations, kernels, strict=True)
        )
        reference_seconds += time.perf_counter() - started
        absolute = abs(fast - reference)
        relative = absolute / max(1.0, abs(reference))
        maximum_absolute_difference = max(maximum_absolute_difference, absolute)
        maximum_relative_difference = max(maximum_relative_difference, relative)
        if absolute > args.absolute_tolerance:
            failures.append(
                {
                    "index": index,
                    "category": labels[index],
                    "fast": fast,
                    "reference": reference,
                    "absolute_difference": absolute,
                }
            )
    result = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "morphology": args.morphology,
        "point_count": args.points,
        "seed": args.seed,
        "unit_points_sha256": unit_sha256,
        "category_counts": {
            name: labels.count(name) for name in ("prior", "central", "edge")
        },
        "absolute_tolerance": args.absolute_tolerance,
        "maximum_absolute_difference": maximum_absolute_difference,
        "maximum_relative_difference": maximum_relative_difference,
        "failure_count": len(failures),
        "first_failures": failures[:20],
        "timing_seconds": {
            "kernels": kernel_seconds,
            "fast_gain_integral": fast_seconds,
            "reference_gain_integral": reference_seconds,
            "estimated_fast_total": kernel_seconds + fast_seconds,
            "estimated_reference_total": kernel_seconds + reference_seconds,
        },
        "estimated_total_speedup": (
            (kernel_seconds + reference_seconds) / (kernel_seconds + fast_seconds)
        ),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.config,
                args.chime_observation,
                args.dsa_observation,
                args.geometry_constraint,
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chime-observation", type=Path, required=True)
    parser.add_argument("--dsa-observation", type=Path, required=True)
    parser.add_argument("--geometry-constraint", type=Path, required=True)
    parser.add_argument("--morphology", choices=("gaussian", "scattering"), required=True)
    parser.add_argument("--points", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
