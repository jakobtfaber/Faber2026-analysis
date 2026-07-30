#!/usr/bin/env python3
"""Resume completed morphology checkpoints and write canonical fit products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fit_one_event_joint_burst import run
from one_event_workflow import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chime-observation", type=Path, required=True)
    parser.add_argument("--dsa-observation", type=Path, required=True)
    parser.add_argument("--geometry-constraint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, require_execution_authorized=True)
    config["_config_path"] = str(args.config.resolve())
    config["paths"]["output_root"] = str(args.output_dir)
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
