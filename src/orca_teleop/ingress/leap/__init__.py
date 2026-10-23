"""Leap Motion (Ultraleap Gemini / LeapC) hand-tracking ingress support."""

__all__ = [
    "leap_hand_to_mediapipe_keypoints",
    "LeapPublisher",
]


def __getattr__(name: str):
    # Lazy so `python -m orca_teleop.ingress.leap.publisher` doesn't trigger
    # the "found in sys.modules ... prior to execution" RuntimeWarning that
    # comes from this package eagerly importing the same module it's later
    # run as __main__ (see ingress.wilor.__init__ for the same pattern).
    if name == "leap_hand_to_mediapipe_keypoints":
        from orca_teleop.ingress.leap.conversion import leap_hand_to_mediapipe_keypoints

        return leap_hand_to_mediapipe_keypoints
    if name == "LeapPublisher":
        from orca_teleop.ingress.leap.publisher import LeapPublisher

        return LeapPublisher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
