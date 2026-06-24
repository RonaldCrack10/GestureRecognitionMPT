

import numpy as np

"""
[0 bis 63] Handform (Normalisiert): Die reinen Positionen der Hand, bereinigt von Kameraabstand und Bildposition.

[63 bis 78] Abstände (15 Werte): Es misst den Abstand zwischen Fingerspitzen (z.B. Daumen zu Zeigefinger). ( "O" oder "C" zu unterscheiden).

[78 bis 93] Beugewinkel (15 Werte): Es berechnet exakt, wie stark jeder einzelne Finger eingeknickt oder ausgestreckt ist. Das ist eine wichtige Info, um ähnliche Buchstaben wie A, E, S und T zu unterscheiden.

[93 bis 156] Geschwindigkeit/Bewegung (63 Werte): Es misst, wie schnell und in welche Richtung sich jeder Punkt im Vergleich zum vorherigen Frame bewegt hat. Das ist überlebenswichtig für dynamische Buchstaben wie J und Z, die eine Bewegung voraussetzen.

"""


WRIST = 0


fingers = [
    (4,  3,  2),   # thumb
    (8,  7,  5),   # index
    (12, 11, 9),   # middle
    (16, 15, 13),  # ring
    (20, 19, 17),  # pinky
]

# Pairs for key inter-landmark distances 
distance_pairs = [
    (4,  8),   
    (4,  12),  
    (4,  16),  
    (4,  20),  
    (8,  12),  
    (8,  16),  
    (12, 16),  
    (16, 20),  
    (5,  17),  
    (0,  9),   
    (8,  5),    
    (12, 9),   
    (16, 13),  
    (20, 17),  
    (4,  0),   
]


def _get_landmarks(pts_flat: np.ndarray) -> np.ndarray:
    """Reshape (T, 63) → (T, 21, 3)."""
    return pts_flat.reshape(len(pts_flat), 21, 3)


def _normalize_coords(pts: np.ndarray) -> np.ndarray:
    """
    Normalisiert die Zeichnung auf [0, 1] basierend auf der 
    Bounding Box aller Fingerspitzen über die gesamte Sequenz.
    """
    lm = _get_landmarks(pts)  # (T, 21, 3)
    T  = lm.shape[0]

    # Alle Fingerspitzen für die Bounding Box verwenden
    TIPS = [4, 8, 12, 16, 20]
    tip_traj = lm[:, TIPS, :2]        # (T, 5, 2) — nur x,y
    tip_traj = tip_traj.reshape(T*5, 2)

    global_min = tip_traj.min(axis=0)  # [x_min, y_min]
    global_max = tip_traj.max(axis=0)  # [x_max, y_max]
    global_range = global_max - global_min
    max_span = np.maximum(global_range.max(), 1e-6)

    # Alle Landmarks relativ zur Bounding Box verschieben
    for t in range(T):
        lm[t, :, :2] = (lm[t, :, :2] - global_min) / max_span
        # z-Koordinate wrist-relativ lassen
        lm[t, :, 2]  = lm[t, :, 2] - lm[t, 0, 2]

    return lm.reshape(T, 63)

def _inter_landmark_distances(lm: np.ndarray) -> np.ndarray:
    """
    Compute 15 pairwise distances between key landmarks.
    These are rotation-invariant and capture hand shape directly.
    lm: (T, 21, 3)
    returns: (T, 15)
    """
    dists = []
    for (i, j) in distance_pairs:
        d = np.linalg.norm(lm[:, i, :] - lm[:, j, :], axis=1, keepdims=True)
        dists.append(d)
    return np.concatenate(dists, axis=1)  # (T, 15)


def _finger_angles(lm: np.ndarray) -> np.ndarray:
    
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


# def _velocity(coords: np.ndarray) -> np.ndarray:
#     """
#     Frame-to-frame delta of the normalized coords.
#     Captures motion information — critical for dynamic letters like J and Z.
#     Velocity at frame 0 is set to zero (no prior frame).
#     coords: (T, 63)
#     returns: (T, 63)
#     """
#     vel = np.zeros_like(coords)
#     vel[1:] = coords[1:] - coords[:-1]
#     return vel


def _extract_features(pts_flat: np.ndarray) -> np.ndarray:
    """
    Main entry point. For STATIC gestures only (no motion needed).

    Input:  (T, 63)  raw flattened MediaPipe landmarks
    Output: (T, 93) engineered feature vector

    Feature breakdown:
      [  0: 63]  normalized xyz coords          — hand pose
      [ 63: 78]  inter-landmark distances (15)  — shape, rotation-invariant
      [ 78: 93]  finger bend angles (15)        — curl encoding
    """
    coords = pts_flat
    lm = _get_landmarks(coords)
    dists  = _inter_landmark_distances(lm)
    angles = _finger_angles(lm)
    return np.concatenate([coords, dists, angles], axis=1)

