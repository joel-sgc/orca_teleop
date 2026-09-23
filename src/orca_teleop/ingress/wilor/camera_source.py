"""Local-webcam WiLoR frame source.

Captures frames from a local camera, sends them to a remote WiLoR inference
server via ``webpolicy``, and returns this publisher's configured hand's
keypoints. The capture/encode/``webpolicy`` client pattern here mirrors a
working reference client for a different, WiLoR-driven robotic hand —
everything in that reference beyond getting keypoints out of WiLoR (its own
hand's joint/motor mapping) is intentionally not carried over; orca_teleop's
own retargeter takes it from there.

Requires ``opencv-python`` and ``webpolicy`` (not orca_teleop dependencies —
install them yourself before using this module).
"""

from __future__ import annotations

import logging

import numpy as np

from orca_teleop.ingress.wilor.publisher import (
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    DEFAULT_JPEG_QUALITY,
    WiLorFrame,
)

logger = logging.getLogger(__name__)


class WiLorCameraSource:
    """Captures webcam frames and queries a remote WiLoR server for keypoints.

    Each call to :meth:`read_frame` is synchronous end-to-end (capture →
    encode → network round-trip to the WiLoR server → parse). That's fine
    here: it's meant to run inside ``WiLorPublisher``'s own background
    reader thread, so nothing else blocks waiting on it, and the round trip
    naturally paces the capture rate to the server's own throughput.
    """

    def __init__(
        self,
        host: str,
        port: int,
        handedness: str = "right",
        camera_index: int | str = 0,
        capture_width: int | None = DEFAULT_CAPTURE_WIDTH,
        capture_height: int | None = DEFAULT_CAPTURE_HEIGHT,
        send_jpeg: bool = False,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        """
        Args:
            send_jpeg: When True, JPEG-encode each frame before sending
                (lower bandwidth, matches a reference client's default).
                Defaults to False (send the raw array via msgpack_numpy)
                because the JPEG path isn't supported by every WiLoR
                server deployment — confirmed empirically against one
                real server, which accepts raw arrays but errors on
                ``{"type": "jpeg"}``. Flip this on only against a server
                you've confirmed handles it.
        """
        import cv2

        self._cv2 = cv2
        self._host = host
        self._port = port
        self._handedness_is_right = handedness.lower() == "right"
        self._send_jpeg = send_jpeg
        self._jpeg_quality = jpeg_quality

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index!r}")
        if capture_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
        if capture_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(capture_height))

        self._client = self._connect()

    def _connect(self):
        from webpolicy.client import Client

        logger.info("Connecting to WiLoR inference server at %s:%s ...", self._host, self._port)
        client = Client(self._host, self._port)
        logger.info("Connected to WiLoR inference server.")
        return client

    def read_frame(self) -> WiLorFrame | None:
        """Capture one frame, query WiLoR, and return this hand's keypoints.

        Returns ``None`` when the configured hand isn't in view this frame.
        Raises on a camera read failure or a WiLoR server error, so
        ``WiLorPublisher``'s reader thread retries with backoff rather than
        this propagating and killing the publisher. A server/connection
        error also rebuilds the websocket client: ``webpolicy.Client``
        doesn't reconnect on its own once its connection is closed, so
        without this every retry after the first failure would keep hitting
        the same dead connection forever.
        """
        cv2 = self._cv2
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the camera")

        if self._send_jpeg:
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            )
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            obs = {"image": buf.tobytes(), "type": "jpeg"}
        else:
            obs = {"image": frame, "type": "image"}

        try:
            response = self._client.step(obs)
        except Exception:
            logger.warning("WiLoR server call failed; reconnecting before the next attempt")
            try:
                self._client = self._connect()
            except Exception:
                logger.exception("WiLoR reconnect failed; will retry on the next read")
            raise

        for hand in response.get("hands") or []:
            if bool(hand.get("is_right")) != self._handedness_is_right:
                continue
            keypoints_3d = hand.get("keypoints_3d")
            if keypoints_3d is None:
                continue
            return WiLorFrame(
                keypoints=np.asarray(keypoints_3d, dtype=np.float32),
                handedness="right" if self._handedness_is_right else "left",
            )
        return None

    def close(self) -> None:
        self._cap.release()
