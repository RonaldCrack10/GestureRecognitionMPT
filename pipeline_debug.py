import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from feature_engineering import _extract_features

TARGET_FRAMES = 65

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def load_letter(label_dir):
    seqs = []
    for npy in sorted(label_dir.glob("*.npy")):
        pts = np.load(npy)
        if len(pts) < 15:
            continue
        seq = pts.reshape(len(pts), -1)
        f = interp1d(np.linspace(0,1,len(seq)), seq, axis=0, 
                     kind='linear', fill_value="extrapolate")
        seq = f(np.linspace(0,1,TARGET_FRAMES)).astype(np.float32)
        seq = _extract_features(seq)
        seqs.append(seq)
    return seqs

data_dir = Path("data")
letters = sorted([d for d in data_dir.iterdir() if d.is_dir()])

cols = 4
rows = (len(letters) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
axes = axes.flatten()

for i, letter_dir in enumerate(letters):
    ax = axes[i]
    seqs = load_letter(letter_dir)
    label = letter_dir.name

    for seq in seqs:
        # coords: seq[:, 0:63] → (65, 21, 3)
        coords = seq[:, :63].reshape(TARGET_FRAMES, 21, 3)
        mid = TARGET_FRAMES // 2
        frame = coords[mid]  # (21, 3) — mittlerer Frame

        # Skelett zeichnen
        for a, b in CONNECTIONS:
            ax.plot([frame[a, 0], frame[b, 0]],
                    [frame[a, 1], frame[b, 1]],
                    'b-', linewidth=0.5, alpha=0.2)

        # Landmarks als Punkte
        ax.scatter(frame[:, 0], frame[:, 1],
                   s=6, alpha=0.3, zorder=5)

    ax.set_title(f"{label} ({len(seqs)})", fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Alle Buchstaben — Handform (mittlerer Frame)", fontsize=13)
plt.tight_layout()
plt.show()