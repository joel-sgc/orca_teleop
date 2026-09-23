"""Drive the physical ORCA hand from an external WiLoR publisher.

Starts the gRPC ingress and retargeter and hands the main thread to the
default sink — the real ``OrcaHand`` (``orca_teleop.pipeline.run()``'s
``sink=None`` default) — then waits for a publisher to connect. Unlike
``teleop_urdf.py``/``teleop_sim.py`` this never launches its own publisher:
run ``orca_teleop.ingress.wilor.publisher`` separately (same machine or a
different one) pointed at this process's --port.

Example:
    # Terminal A: this script, drives the real hand
    python scripts/teleop_wilor.py \\
        --model-path /path/to/orcahand-right/config.yaml

    # Terminal B: feeds it hand tracking from a WiLoR server
    python -m orca_teleop.ingress.wilor.publisher \\
        --server localhost:50051 --hand right \\
        --wilor-host <wilor-server-host> --wilor-port 8084
"""

from __future__ import annotations

import argparse
import logging

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the physical ORCA hand from an external publisher (e.g. WiLoR)."
    )
    parser.add_argument("--model-path", default=None, help="OrcaHand config.yaml path")
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config",
        default=None,
        help="YAML config for --retargeter adaptive_analytical",
    )
    parser.add_argument(
        "--visualize-landmarks",
        action="store_true",
        help="Open a live 3D matplotlib window showing hand keypoints",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        port=args.port,
        visualize_landmarks=args.visualize_landmarks,
        retargeter_backend=args.retargeter,
        retargeter_config_path=args.retarget_config,
    )


if __name__ == "__main__":
    main()
