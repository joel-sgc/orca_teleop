"""Leap Motion (Ultraleap Gemini / LeapC) gRPC publisher.

Run this on the machine with a Leap Motion device plugged in (any generation
Ultraleap Gemini still supports, including the original 2013 controller —
see https://support.ultraleap.com/hc/en-us/articles/15504099024541). It
subscribes to Ultraleap's tracking service via the ``leap`` package's
listener API, converts hand tracking to the pipeline's canonical MediaPipe
order, and streams it to the robot-side ``IngressServer`` over gRPC — same
shape as the MediaPipe/Manus/WiLoR publishers.

Requires the Ultraleap Gemini Hand Tracking service installed and running
locally, plus this project's 'leap' extra (not a plain pip/PyPI install —
see pyproject.toml's [tool.uv.sources] and the 'leap' extra's own comment).

Usage::

    # Stream to a robot on the same machine
    python -m orca_teleop.ingress.leap.publisher

    # Stream to a remote robot, left hand
    python -m orca_teleop.ingress.leap.publisher --server 192.168.1.42:50051 --hand left
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import grpc

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.leap.conversion import leap_hand_to_mediapipe_keypoints

logger = logging.getLogger(__name__)

_POLL_PERIOD_S = 0.001
_GRPC_RETRY_INITIAL_S = 0.5
_GRPC_RETRY_MAX_S = 10.0


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


class LeapPublisher:
    """Streams Leap Motion hand tracking to IngressServer over gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
    ) -> None:
        self._server_address = server_address
        self._handedness = handedness.lower()

        self._lock = threading.Lock()
        self._latest_keypoints = None
        self._fresh = False
        self._stop = threading.Event()
        self._hand_visible = False
        self._last_seen_sides: list[str] | None = None

    def _on_tracking_event(self, event) -> None:
        """Leap listener callback — fires on the SDK's own background thread."""
        found = False
        seen_sides = []
        for hand in event.hands:
            hand_side = "right" if str(hand.type) == "HandType.Right" else "left"
            seen_sides.append(hand_side)
            if hand_side != self._handedness:
                continue

            found = True
            keypoints = leap_hand_to_mediapipe_keypoints(hand)
            with self._lock:
                self._latest_keypoints = keypoints
                self._fresh = True

        if seen_sides != self._last_seen_sides:
            self._last_seen_sides = seen_sides
            logger.info("Leap sees hands: %s (want: %s)", seen_sides, self._handedness)

        if found != self._hand_visible:
            self._hand_visible = found
            logger.info("%s hand %s", self._handedness, "found" if found else "lost")

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

    def run(self) -> None:
        """Open the Leap connection, connect to the server, and stream until interrupted."""
        import leap

        on_tracking_event = self._on_tracking_event

        class _Listener(leap.Listener):
            def on_connection_event(self, event):
                logger.info("Connected to Leap tracking service.")

            def on_device_event(self, event):
                try:
                    with event.device.open():
                        info = event.device.get_info()
                except leap.LeapCannotOpenDeviceError:
                    info = event.device.get_info()
                logger.info("Found Leap device %s", info.serial)

            def on_tracking_event(self, event):
                on_tracking_event(event)

        connection = leap.Connection()
        connection.add_listener(_Listener())

        logger.info(
            "LeapPublisher starting (server=%s, hand=%s)",
            self._server_address,
            self._handedness,
        )

        backoff = _Backoff(_GRPC_RETRY_INITIAL_S, _GRPC_RETRY_MAX_S)
        try:
            with connection.open():
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
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
                            self._server_address,
                            e.code() if hasattr(e, "code") else e,
                            backoff.current_s,
                        )
                        backoff.sleep_and_grow()
                    finally:
                        channel.close()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            logger.info("LeapPublisher shut down.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Leap Motion hand tracking to the orca_teleop server via gRPC.",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    publisher = LeapPublisher(server_address=args.server, handedness=args.hand)
    publisher.run()


if __name__ == "__main__":
    main()
