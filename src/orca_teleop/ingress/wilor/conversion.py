"""WiLoR coordinate conversion for the teleop pipeline.

All WiLoR-specific joint handling lives here so the rest of the pipeline
(retargeter, sim) stays source-agnostic, mirroring
``orca_teleop.ingress.manus.conversion``.
"""

from __future__ import annotations

import numpy as np

# Confirmed directly against this deployment's raw output (plotted and
# visually matched by hand): WiLoR/wilor_mini already emits joints in
# MediaPipe order — 0=wrist, then each finger's own points *including its
# tip*, as thumb (1-4), index (5-8), middle (9-12), ring (13-16), pinky
# (17-20) — matching ``retargeting.utils.get_mano_joints_dict`` directly.
# This is identity, not a permutation; it's kept as an explicit list (rather
# than deleting the reorder step entirely) so a future WiLoR/wilor_mini
# version that changes its output order has one obvious place to fix.
WILOR_TO_MEDIAPIPE = list(range(21))


def wilor_to_mediapipe_keypoints(joints: np.ndarray) -> np.ndarray:
    """Convert WiLoR's 21 joints (already MediaPipe-ordered) to wrist-relative.

    No axis remap is applied: the retargeter's normalization step
    (``get_normalized_local_manohand_joint_pos``) derives its own coordinate
    frame from hand geometry every frame, so translation doesn't need
    correcting here either — this recentering is just a defensive no-op
    given the source already reports wrist-relative coordinates (see
    ``ingress.manus.conversion.manus_zmq_to_mano_keypoints`` for the same
    reasoning applied to another source).

    Args:
        joints: ``(21, 3)`` array in WiLoR joint order, any consistent unit
            and origin.

    Returns:
        ``(21, 3)`` float32 array in MediaPipe order, wrist-relative.
    """
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape != (21, 3):
        raise ValueError(f"joints must have shape (21, 3); got {joints.shape}")

    keypoints = joints[WILOR_TO_MEDIAPIPE]
    return keypoints - keypoints[0:1, :]
