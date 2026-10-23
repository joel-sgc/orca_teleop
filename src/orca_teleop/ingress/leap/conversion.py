"""Leap Motion (Ultraleap Gemini / LeapC) coordinate conversion.

All Leap-specific joint handling lives here so the rest of the pipeline
(retargeter, sim) stays source-agnostic, mirroring
``orca_teleop.ingress.manus.conversion`` and
``orca_teleop.ingress.wilor.conversion``.
"""

from __future__ import annotations

import numpy as np

MM_TO_M = 0.001

# A leap.datatypes.Hand exposes digits in thumb, index, middle, ring, pinky
# order (`hand.digits == [hand.thumb, hand.index, hand.middle, hand.ring,
# hand.pinky]`), and each Digit has 4 Bones — metacarpal, proximal,
# intermediate, distal — base to tip. Walking each digit's 4 bones via
# bone.next_joint, in that finger order, lines up directly with MediaPipe's
# point layout: wrist, then each finger's own points including its tip, as
# thumb, index, middle, ring, pinky. No reorder permutation needed (unlike
# WiLoR/MANO's own joint order) — confirmed against Ultraleap's own
# leapc-python-bindings examples (simple_pinching_example.py's
# ``hand.digits[digit_idx].distal.next_joint`` usage) and datatypes.py.
_BONES_PER_DIGIT = 4


def leap_hand_to_mediapipe_keypoints(hand) -> np.ndarray:
    """Convert a ``leap.datatypes.Hand`` to wrist-relative MediaPipe-order keypoints.

    No axis remap is applied: the retargeter's normalization step
    (``get_normalized_local_manohand_joint_pos``) derives its own coordinate
    frame from hand geometry every frame, so only unit conversion (Leap
    reports millimeters) and point order matter here (see
    ``ingress.manus.conversion.manus_zmq_to_mano_keypoints`` for the same
    reasoning applied to another source).

    Args:
        hand: A ``leap.datatypes.Hand`` from a Leap ``TrackingEvent``.

    Returns:
        ``(21, 3)`` float32 array in MediaPipe order, wrist-relative, meters.
    """
    keypoints = np.zeros((21, 3), dtype=np.float32)

    wrist = hand.arm.next_joint  # the wrist end of the forearm "bone"
    keypoints[0] = (wrist.x, wrist.y, wrist.z)

    for finger_idx, digit in enumerate(hand.digits):
        for bone_idx in range(_BONES_PER_DIGIT):
            joint = digit.bones[bone_idx].next_joint
            point_idx = 1 + finger_idx * _BONES_PER_DIGIT + bone_idx
            keypoints[point_idx] = (joint.x, joint.y, joint.z)

    keypoints *= MM_TO_M
    return keypoints - keypoints[0:1, :]
