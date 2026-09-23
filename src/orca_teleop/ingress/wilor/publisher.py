"""WiLoR gRPC publisher.

Reads hand keypoints from a WiLoR-based tracking server and streams them to
the robot-side ``IngressServer`` over gRPC, same as the MediaPipe and Manus
publishers. WiLoR inference runs elsewhere (typically its own GPU machine,
reached over the network via ``webpolicy``); this publisher captures the
local webcam, sends frames to it, and forwards the returned keypoints — it
doesn't run any inference itself.

Usage::

    # Capture a local webcam, query a remote WiLoR server, stream to a robot
    python -m orca_teleop.ingress.wilor.publisher \\
        --server 192.168.1.42:50051 --hand right \\
        --wilor-host 147.126.2.125 --wilor-port 8084

    # Bring your own frame source instead (skips WiLorCameraSource/webpolicy
    # entirely — useful if frames come from somewhere else)
    python -m orca_teleop.ingress.wilor.publisher \\
        --server 192.168.1.42:50051 --read-frame mypackage.mymodule:read_frame

``--read-frame`` (a ``module.path:attribute`` spec resolving to a
zero-argument callable returning a :class:`WiLorFrame` or ``None``) takes
priority over ``--wilor-host``/``--wilor-port`` when both are given.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import grpc
import numpy as np

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.wilor.conversion import wilor_to_mediapipe_keypoints

logger = logging.getLogger(__name__)

_POLL_PERIOD_S = 0.001
_READ_RETRY_INITIAL_S = 0.2
_READ_RETRY_MAX_S = 5.0
_GRPC_RETRY_INITIAL_S = 0.5
_GRPC_RETRY_MAX_S = 10.0

# Defaults for WiLorCameraSource, kept here (rather than in camera_source.py,
# which needs WiLorFrame from this module) to avoid a circular import.
DEFAULT_JPEG_QUALITY = 75
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480


@dataclass
class WiLorFrame:
    """One frame read from the WiLoR source.

    Attributes:
        keypoints: ``(21, 3)`` array of the 21 MANO joints, already trimmed
            down from whatever WiLoR itself returns (which may include
            extra points) — see ``ingress.wilor.conversion`` for the
            MANO-order assumption this publisher converts from. Any
            consistent unit/origin; no axis remap is applied.
        handedness: ``"left"`` or ``"right"`` if the source reports it.
            When set and it doesn't match the publisher's configured hand,
            the frame is dropped. When ``None``, every frame is forwarded
            under the publisher's configured hand.
    """

    keypoints: np.ndarray
    handedness: str | None = None


WiLorFrameSource = Callable[[], WiLorFrame | None]


class _Backoff:
    """Exponential backoff, reset after a success."""

    def __init__(self, initial_s: float, max_s: float) -> None:
        self._initial_s = initial_s
        self._max_s = max_s
        self._current_s = initial_s

    @property
    def current_s(self) -> float:
        return self._current_s

    def reset(self) -> None:
        self._current_s = self._initial_s

    def sleep_and_grow(self) -> None:
        time.sleep(self._current_s)
        self._current_s = min(self._current_s * 2, self._max_s)


class WiLorPublisher:
    """Reads WiLoR frames via ``read_frame`` and streams them to IngressServer."""

    def __init__(
        self,
        server_address: str,
        handedness: str,
        read_frame: WiLorFrameSource,
    ) -> None:
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._read_frame = read_frame

        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None
        self._fresh = False
        self._stop = threading.Event()

    def _source_reader(self) -> None:
        """Background thread: poll ``read_frame``, reconnecting on failure.

        A transient error talking to the WiLoR server (network hiccup, the
        server restarting) is logged and retried with backoff rather than
        killing the publisher; only ``stop()`` ends this loop.
        """
        backoff = _Backoff(_READ_RETRY_INITIAL_S, _READ_RETRY_MAX_S)
        while not self._stop.is_set():
            try:
                frame = self._read_frame()
            except Exception:
                logger.exception(
                    "WiLoR source read failed; retrying in %.1fs", backoff.current_s
                )
                backoff.sleep_and_grow()
                continue

            backoff.reset()
            if frame is None:
                time.sleep(_POLL_PERIOD_S)
                continue
            if frame.handedness is not None and frame.handedness.lower() != self._handedness:
                continue

            keypoints = wilor_to_mediapipe_keypoints(frame.keypoints)
            with self._lock:
                self._latest_keypoints = keypoints
                self._fresh = True

    def _frame_generator(self):
        """Yield HandFrame protos as fast as new data arrives."""
        while not self._stop.is_set():
            with self._lock:
                if not self._fresh:
                    kp = None
                else:
                    kp = self._latest_keypoints.copy()
                    self._fresh = False

            if kp is None:
                time.sleep(_POLL_PERIOD_S)
                continue

            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness=self._handedness,
                timestamp_ns=time.time_ns(),
            )

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Start the source-reader thread and stream to the robot until stopped.

        The gRPC connection itself is reconnected with backoff on failure —
        a dropped stream (robot-side restart, network blip) is logged and
        retried rather than ending the process.
        """
        logger.info(
            "WiLorPublisher starting (server=%s, hand=%s)",
            self._server_address,
            self._handedness,
        )

        reader_thread = threading.Thread(
            target=self._source_reader, name="wilor-source-reader", daemon=True
        )
        reader_thread.start()

        backoff = _Backoff(_GRPC_RETRY_INITIAL_S, _GRPC_RETRY_MAX_S)
        try:
            while not self._stop.is_set():
                channel = grpc.insecure_channel(self._server_address)
                stub = hand_stream_pb2_grpc.HandStreamStub(channel)
                stream_future = stub.StreamHandFrames.future(self._frame_generator())
                try:
                    stream_future.result()
                    backoff.reset()
                except grpc.RpcError as e:
                    if self._stop.is_set():
                        break
                    logger.warning(
                        "gRPC stream to %s dropped (%s); reconnecting in %.1fs",
                        self._server_address, e.code() if hasattr(e, "code") else e,
                        backoff.current_s,
                    )
                    backoff.sleep_and_grow()
                finally:
                    channel.close()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            reader_thread.join(timeout=2.0)
            logger.info("WiLorPublisher shut down.")


def _unconfigured_read_frame() -> WiLorFrame | None:
    raise RuntimeError(
        "Neither --read-frame nor --wilor-host was given; there's no WiLoR "
        "source to read from. See WiLorPublisher's module docstring."
    )


def _load_read_frame(spec: str) -> WiLorFrameSource:
    """Resolve a ``module.path:attribute`` spec into a callable."""
    module_path, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--read-frame must be 'module.path:attribute'; got {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _build_read_frame(args: argparse.Namespace) -> WiLorFrameSource:
    """Resolve the configured frame source: an explicit --read-frame spec
    takes priority; otherwise a --wilor-host/--wilor-port pair builds a
    WiLorCameraSource; otherwise fail fast with instructions."""
    if args.read_frame is not None:
        return _load_read_frame(args.read_frame)
    if args.wilor_host is not None:
        from orca_teleop.ingress.wilor.camera_source import WiLorCameraSource

        source = WiLorCameraSource(
            host=args.wilor_host,
            port=args.wilor_port,
            handedness=args.hand,
            camera_index=args.camera,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            send_jpeg=args.send_jpeg,
            jpeg_quality=args.jpeg_quality,
        )
        return source.read_frame
    return _unconfigured_read_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream WiLoR hand keypoints to the orca_teleop server via gRPC.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Which hand to stream (default: right)",
    )
    parser.add_argument(
        "--wilor-host",
        default=None,
        help="Address of the remote WiLoR inference server (webpolicy host).",
    )
    parser.add_argument(
        "--wilor-port",
        type=int,
        default=8084,
        help="Port of the remote WiLoR inference server (default: 8084).",
    )
    parser.add_argument(
        "--camera",
        default=0,
        help="Local camera index or path (default: 0). Only used with --wilor-host.",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=DEFAULT_CAPTURE_WIDTH,
        help=f"Requested capture width (default: {DEFAULT_CAPTURE_WIDTH}).",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=DEFAULT_CAPTURE_HEIGHT,
        help=f"Requested capture height (default: {DEFAULT_CAPTURE_HEIGHT}).",
    )
    parser.add_argument(
        "--send-jpeg",
        action="store_true",
        help=(
            "JPEG-encode frames before sending (lower bandwidth). Off by "
            "default: not every WiLoR server deployment supports it — "
            "confirm yours does before enabling this."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=(
            f"JPEG quality, 1-100, only used with --send-jpeg "
            f"(default: {DEFAULT_JPEG_QUALITY})."
        ),
    )
    parser.add_argument(
        "--read-frame",
        default=None,
        help=(
            "'module.path:attribute' resolving to a zero-argument callable "
            "returning a WiLorFrame (or None). Bypasses --wilor-host/"
            "WiLorCameraSource entirely when given."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    try:
        args.camera = int(args.camera)
    except ValueError:
        pass  # a device path (e.g. /dev/video0) rather than an index

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    publisher = WiLorPublisher(
        server_address=args.server,
        handedness=args.hand,
        read_frame=_build_read_frame(args),
    )
    publisher.run()


if __name__ == "__main__":
    main()
