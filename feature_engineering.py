"""
feature_engineering.py
Drop-in replacement for the _normalize + feature extraction in labeling.py.

Current baseline: 63 raw xyz coords per frame
New feature vector: 63 + 15 + 15 + 63 = 156 dims per frame

How to use
----------
Replace _normalize(seq) in dataset_building() with _extract_features(seq):

    seq = pts.reshape(len(pts), -1)      # (T, 63)
    seq = _extract_features(seq)         # (T, 156)  ← was _normalize(seq)
"""

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0

# Fingertip and base landmarks for each finger
#           tip   pip   mcp
FINGERS = [
    (4,  3,  2),   # thumb
    (8,  7,  5),   # index
    (12, 11, 9),   # middle
    (16, 15, 13),  # ring
    (20, 19, 17),  # pinky
]

# Pairs for key inter-landmark distances (captures hand shape)
DISTANCE_PAIRS = [
    (4,  8),   # thumb tip  ↔ index tip   (pinch)
    (4,  12),  # thumb tip  ↔ middle tip
    (4,  16),  # thumb tip  ↔ ring tip
    (4,  20),  # thumb tip  ↔ pinky tip
    (8,  12),  # index tip  ↔ middle tip  (V/U discrimination)
    (8,  16),  # index tip  ↔ ring tip
    (12, 16),  # middle tip ↔ ring tip
    (16, 20),  # ring tip   ↔ pinky tip
    (5,  17),  # index mcp  ↔ pinky mcp   (palm width — scale ref)
    (0,  9),   # wrist      ↔ middle mcp  (palm height — scale ref)
    (8,  5),   # index tip  ↔ index mcp   (finger extension)
    (12, 9),   # middle tip ↔ middle mcp
    (16, 13),  # ring tip   ↔ ring mcp
    (20, 17),  # pinky tip  ↔ pinky mcp
    (4,  0),   # thumb tip  ↔ wrist
]


def _get_landmarks(pts_flat: np.ndarray) -> np.ndarray:
    """Reshape (T, 63) → (T, 21, 3)."""
    return pts_flat.reshape(len(pts_flat), 21, 3)


def _normalize_coords(pts: np.ndarray) -> np.ndarray:
    """
    Normalize (T, 63) coords:
      - subtract wrist (landmark 0) to center
      - scale by hand span (distance between index MCP #5 and pinky MCP #17)
        This is more stable than max-norm because it's a consistent anatomical
        distance rather than being driven by whichever landmark happens to be
        furthest away in a given frame.
    """
    lm = _get_landmarks(pts)                        # (T, 21, 3)
    wrist = lm[:, WRIST:WRIST+1, :]                 # (T, 1, 3)
    lm = lm - wrist                                 # center on wrist

    # Palm width as scale reference (more stable than max-norm)
    palm_width = np.linalg.norm(
        lm[:, 5, :] - lm[:, 17, :], axis=1, keepdims=True  # (T, 1)
    )
    palm_width = np.maximum(palm_width, 1e-6)[:, :, np.newaxis]  # (T, 1, 1)
    lm = lm / palm_width

    return lm.reshape(len(pts), 63)


def _inter_landmark_distances(lm: np.ndarray) -> np.ndarray:
    """
    Compute 15 pairwise distances between key landmarks.
    These are rotation-invariant and capture hand shape directly.
    lm: (T, 21, 3)
    returns: (T, 15)
    """
    dists = []
    for (i, j) in DISTANCE_PAIRS:
        d = np.linalg.norm(lm[:, i, :] - lm[:, j, :], axis=1, keepdims=True)
        dists.append(d)
    return np.concatenate(dists, axis=1)  # (T, 15)


def _finger_angles(lm: np.ndarray) -> np.ndarray:
    """
    Compute bend angle at the PIP joint for each of the 5 fingers (15 values:
    3 angles per finger — MCP, PIP, DIP).
    Angle = arccos of the dot product of the two bone vectors.
    These directly encode finger curl, which differentiates many ASL pairs
    (e.g. E vs A, B vs 4, S vs A).
    lm: (T, 21, 3)
    returns: (T, 15)
    """
    # Full joint chains per finger: [mcp, pip, dip, tip]
    chains = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index
        [9, 10, 11, 12], # middle
        [13, 14, 15, 16],# ring
        [17, 18, 19, 20],# pinky
    ]
    angles = []
    for chain in chains:
        for k in range(len(chain) - 2):  # 3 angles per finger
            a = lm[:, chain[k], :]
            b = lm[:, chain[k+1], :]
            c = lm[:, chain[k+2], :]
            v1 = a - b
            v2 = c - b
            cos_a = np.einsum('ti,ti->t', v1, v2) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angles.append(np.arccos(cos_a)[:, np.newaxis])
    return np.concatenate(angles, axis=1)  # (T, 15)


def _velocity(coords: np.ndarray) -> np.ndarray:
    """
    Frame-to-frame delta of the normalized coords.
    Captures motion information — critical for dynamic letters like J and Z.
    Velocity at frame 0 is set to zero (no prior frame).
    coords: (T, 63)
    returns: (T, 63)
    """
    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    return vel


def _extract_features(pts_flat: np.ndarray) -> np.ndarray:
    """
    Main entry point. Replaces _normalize() in dataset_building().

    Input:  (T, 63)  raw flattened MediaPipe landmarks
    Output: (T, 156) engineered feature vector

    Feature breakdown:
      [  0: 63]  normalized xyz coords          — hand pose
      [ 63: 78]  inter-landmark distances (15)  — shape, rotation-invariant
      [ 78: 93]  finger bend angles (15)        — curl encoding
      [ 93:156]  frame-to-frame velocity (63)   — motion (key for J, Z)
    """
    coords = _normalize_coords(pts_flat)
    lm = _get_landmarks(coords)
    dists  = _inter_landmark_distances(lm)
    angles = _finger_angles(lm)
    vel    = _velocity(coords)
    return np.concatenate([coords, dists, angles, vel], axis=1)

