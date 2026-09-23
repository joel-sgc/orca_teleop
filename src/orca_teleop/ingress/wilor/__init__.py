"""WiLoR hand-tracking ingress support."""

__all__ = [
    "WILOR_TO_MEDIAPIPE",
    "wilor_to_mediapipe_keypoints",
    "WiLorCameraSource",
    "WiLorFrame",
    "WiLorFrameSource",
    "WiLorPublisher",
]


def __getattr__(name: str):
    # Lazy so `python -m orca_teleop.ingress.wilor.publisher` doesn't trigger
    # the "found in sys.modules ... prior to execution" RuntimeWarning that
    # comes from this package eagerly importing the same module it's later
    # run as __main__.
    if name in ("WILOR_TO_MEDIAPIPE", "wilor_to_mediapipe_keypoints"):
        from orca_teleop.ingress.wilor import conversion

        return getattr(conversion, name)
    if name == "WiLorCameraSource":
        from orca_teleop.ingress.wilor.camera_source import WiLorCameraSource

        return WiLorCameraSource
    if name in ("WiLorFrame", "WiLorFrameSource", "WiLorPublisher"):
        from orca_teleop.ingress.wilor import publisher

        return getattr(publisher, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
