import pickle
import numpy as np
from pathlib import Path

def convert_recording(pkl_path: Path, save_path: Path, min_frames: int = 15):
    with open(pkl_path, "rb") as f:
        recording = pickle.load(f)

    frames = []
    for entry in recording["detector"]:
        result = entry.get("detector")
        if result is None or not result.hand_landmarks:
            continue
        landmarks = result.hand_landmarks[0]  # erste Hand
        frame = [[lm.x, lm.y, lm.z] for lm in landmarks]  # (21, 3)
        frames.append(frame)

    if len(frames) < min_frames:
        print(f"  ✗ {pkl_path.name}: nur {len(frames)} Frames — übersprungen")
        return False

    arr = np.array(frames, dtype=np.float32)  # (T, 21, 3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, arr)
    print(f"  ✓ {pkl_path.name}: {len(frames)} Frames → {save_path}")
    return True

recordings_dir = Path("C:/Users/rashi/Desktop/GestureRecognitionMPT-recordings-v1/GestureRecognitionMPT-recordings-v1/recordings/recordings")

for label_dir in sorted(recordings_dir.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    existing = len(list((Path("data") / label).glob("*.npy")))

    for i, pkl in enumerate(sorted(label_dir.glob("*.pkl"))):
        save_path = Path("data") / label / f"{label}_{existing + i + 1:03d}.npy"
        convert_recording(pkl, save_path)