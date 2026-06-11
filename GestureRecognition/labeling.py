import sys
import time
import argparse
import subprocess
import numpy as np
import msvcrt
from pathlib import Path
from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules.handdetector import HandDetector
from GestureRecognition.modules.trailmarker import TrailMarker
from GestureRecognition.modules.datarecorder import DataRecorder, _StopRecording
from feature_engineering import _extract_features
import pickle
from scipy.interpolate import interp1d
 
 
 
def _normalize(raw_traj: np.ndarray) -> np.ndarray: 
    """
    Normalisiert die Punkte durch frame-basiertes Zentrieren und Skalieren.
    Ersetzt das alte np.tile-Verfahren für exakte Übereinstimmung mit dem Preprocessor.
    """
    T = raw_traj.shape[0]
    # In (T, 21, 3) umwandeln für saubere Indizierung
    reshaped_traj = raw_traj.reshape(T, 21, 3)
    
    # Handgelenk (Landmark 0) abziehen
    wrist = reshaped_traj[:, 0, :][:, np.newaxis, :]
    normalized_traj = reshaped_traj - wrist

    # Jedes Frame einzeln auf seine maximale Landmark-Distanz skalieren
    for t in range(T):
        frame_dist = np.linalg.norm(normalized_traj[t], axis=1).max()
        if frame_dist > 1e-6:
            normalized_traj[t] /= frame_dist

    # Zurückdrehen zu (T, 63)
    return normalized_traj.reshape(T, 63)


def _resample_trajectory(traj: np.ndarray, target_frames: int = 30) -> np.ndarray:
    """
    Interpoliert die Sequenz linear auf eine feste Anzahl von Frames (Zeitschritte).
    Verhindert das Driften von HMM Log-Likelihood-Scores bei unterschiedlichen Geschwindigkeiten.
    """
    T = traj.shape[0]
    if T == target_frames:
        return traj
        
    current_time_grid = np.linspace(0, 1, T)
    target_time_grid = np.linspace(0, 1, target_frames)
    
    f = interp1d(current_time_grid, traj, axis=0, kind='linear', fill_value="extrapolate")
    return f(target_time_grid).astype(np.float32)

 
def data_labeling(times: int, label: str, finger_idx: int = 8):
    """
    Startet für jede Aufnahme einen eigenen Subprocess.
    ESC speichert die Aufnahme als .npy und beendet das Fenster.
 
    Parameters
    ----------
    times : int
        Anzahl der gewünschten Aufnahmen.
    label : str
        Name der Geste (Unterordner unter data/).
    finger_idx : int
        MediaPipe-Landmark-Index (Standard 8 = Zeigefingerspitze).
    """
    save_dir = Path("data") / label
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.npy")))
 
    saved = 0
    while saved < times:
        idx = existing + saved + 1
        save_path = save_dir / f"{label}_{idx:03d}.npy"
        print(f"\n[{label}]  Aufnahme {saved + 1}/{times}")
        print(f"  Datei: {save_path}")
 
        # Diese Datei ruft sich selbst mit --single auf
        subprocess.run([
            sys.executable, __file__,
            "--single",
            "--save-path",  str(save_path),
            "--finger-idx", str(finger_idx),
        ])
 
        # Qt braucht Zeit zum Aufräumen bevor ein neues Fenster geöffnet werden kann
        time.sleep(2)
 
        if save_path.exists():
            saved += 1
        else:
            print("  ✗ Nicht gespeichert — nochmal versuchen? (j / andere Taste = abbrechen)")
            while msvcrt.kbhit():
                msvcrt.getch()
            if msvcrt.getch().lower() != b"j":
                print("  Abgebrochen.")
                break
 
    print(f"\nFertig: {saved}/{times} Aufnahmen für '{label}' gespeichert.")
 
 

 
def _single_recording(save_path: Path, finger_idx: int): # Diese Funktion wird in einem Subprocess aufgerufen, um eine einzelne Aufnahme zu machen. Sie startet die Engine mit einem DataRecorder, der auf ESC wartet, um die Aufnahme zu speichern und die Engine zu stoppen.  
    

    recorder = DataRecorder(save_path=save_path, finger_idx=finger_idx)

    parser = argparse.ArgumentParser("GestureRecognition")
    parser.add_argument("--mode",        action="store",      default="none")
    parser.add_argument("--single",      action="store_true", default=False)  
    parser.add_argument("--save-path",   type=str)                            
    parser.add_argument("--finger-idx",  type=int,            default=8)      

    modules = [
        ConfigParser(parser),
        Webcam(),
        HandDetector(),
        TrailMarker(),
        recorder,
    ]

    engine = Engine(modules=modules, signals={})

    try:
        engine.run({})
    except (_StopRecording, SystemExit, Exception):
        pass
 
 
 
def dataset_building(output_path):
    """
    Lädt alle .npy-Aufnahmen aus data/<label>/,
    extrahiert Features und speichert einen hmmlearn-kompatiblen Datensatz.

    Parameters
    ----------
    output_path : str or Path
        Zielpfad für die dataset.pickle-Datei.
    """

    data_dir    = Path("data")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MIN_FRAMES = 15
    X, lengths, labels, classes = [], [], [], []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or not any(label_dir.glob("*.npy")):
            continue

        label = label_dir.name
        classes.append(label)
        count = 0

        for npy in sorted(label_dir.glob("*.npy")):
            pts = np.load(npy)               # (T, 21, 3)
            seq = pts.reshape(len(pts), -1)  # (T, 63)

            if len(seq) < MIN_FRAMES:
                continue

            seq = _resample_trajectory(seq, target_frames=58)  # (30, 63)
            seq = _extract_features(seq)                        # (30, 156)  ← NEU

            X.append(seq)
            lengths.append(len(seq))
            labels.append(label)
            count += 1
            print(f"  ✓ {label}/{npy.name}: {len(seq)} Frames")

        print(f"→ '{label}': {count} Sequenzen\n")

    if not X:
        print("Dataset leer — keine Daten gefunden.")
        return

   

    dataset = {
        "X": np.concatenate(X),
        "lengths": lengths,
        "labels":  labels,
        "classes": classes,
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset gespeichert → {output_path}")
    print(f"{len(classes)} Klassen · {len(lengths)} Sequenzen · {len(np.concatenate(X))} Frames gesamt")
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser("labeling")
    parser.add_argument("--single",      action="store_true", default=False)
    parser.add_argument("--save-path",   type=str)
    parser.add_argument("--finger-idx",  type=int, default=8)
    args, _ = parser.parse_known_args()
 
    if args.single:
        # Subprocess-Modus: eine einzelne Aufnahme
        _single_recording(
            save_path  = Path(args.save_path),
            finger_idx = args.finger_idx,
        )
    else:
        # Normaler Modus
        GESTEN    = ["B", "P"]     # ← Gesten anpassen
        AUFNAHMEN = 10       # ← Anzahl Aufnahmen pro Geste
 
        for geste in GESTEN:
            data_labeling(times=AUFNAHMEN, label=geste)
 
        dataset_building("data/dataset.pickle")