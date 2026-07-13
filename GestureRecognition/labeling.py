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
import pickle
from scipy.interpolate import interp1d
 
 
 
 
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
    existing = list(save_dir.glob("*.npy"))
    if existing:
        highest = max(
            int(f.stem.split("_")[-1])
            for f in existing
        )
    else:
        highest = 0
            

 
    saved = 0
    while saved < times:
        idx = highest + saved + 1
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



def _resample(traj: np.ndarray, target_frames: int = 70) -> np.ndarray:  
    
    T = traj.shape[0]
    if T == target_frames:
        return traj
    f = interp1d(
        np.linspace(0, 1, T),
        traj,
        axis=0,
        kind='linear',
        fill_value="extrapolate"
    )
    return f(np.linspace(0, 1, target_frames)).astype(np.float32)
TARGET_FRAMES = 70

def _normalize_trajectory_only(pts_flat: np.ndarray) -> np.ndarray:
    """
    Normalisiert die Sequenz so, dass die Flugbahn der Zeigefingerspitze (Landmark 8)
    perfekt in eine Einheitsbox [0, 1] passt. Die restlichen Hand-Landmarks 
    werden relativ dazu mitverschrumpft/-geweitet, ohne ihre Form zu verlieren.
    """
    T = pts_flat.shape[0] # T ist die Anzahl der Frames in der Sequenz
    lm = pts_flat.reshape(T, 21, 2)
    
   # Extrahiere die Flugbahn NUR von Landmark 8 (Zeigefingerspitze)
   
    tip_trajectory = lm[:, 8, :]
    
    # Finde die Bounding-Box NUR für die Zeichnung der Fingerspitze, das heißt, wir wollen die minimalen und maximalen X- und Y-Koordinaten der Fingerspitze über alle Frames hinweg bestimmen.
    min_coords = tip_trajectory.min(axis=0)  # [min_x, min_y] der Zeichnung
    max_coords = tip_trajectory.max(axis=0)  # [max_x, max_y] der Zeichnung
    
    #Berechne die maximale Ausdehnung der Zeichnung (Breite oder Höhe)
    span = max_coords - min_coords
    max_span = np.maximum(span.max(), 1e-6)  # Verhindert Division durch 0 bei Standbildern
    
    #Wende diese Skalierung auf ALLE 21 Landmarks an
    # Wir ziehen von jedem Punkt das Minimum der Fingerspitze ab und teilen durch die Fingerspitzen-Spanne
    for t in range(T):
        lm[t] = (lm[t] - min_coords) / max_span
        
    
    return lm.reshape(T, 42)
 
 
 
def dataset_building(output_path):
    """
    Lädt alle .npy-Aufnahmen, reduziert sie auf X- und Y-Koordinaten,
    interpoliert sie auf eine feste Frame-Anzahl und speichert das Dataset.
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

        for npy in sorted(label_dir.glob("*.npy")):
            pts = np.load(npy)  # Rohdaten laden

            # 2. Jetzt erst flachklopfen auf (T, 42), da wir nur X- und Y-Koordinaten brauchen (21 Landmarks * 2 Koordinaten)
            pts_flat = pts.reshape(len(pts), 42)

            if len(pts_flat) < MIN_FRAMES:
                print(f"  ✗ {label}/{npy.name}: Zu kurz ({len(pts_flat)} Frames) — übersprungen")
                continue

            pts_normalized = _normalize_trajectory_only(pts_flat)

            #  Auf feste Frame-Anzahl bringen -> Liefert (65, 42)
            seq = _resample(pts_normalized, TARGET_FRAMES) 
            
            X.append(seq)
            lengths.append(len(seq))
            labels.append(label)
            print(f"  ✓ {label}/{npy.name}: {len(seq)} Frames")

    if not X:
        print("Dataset leer — keine Daten gefunden.")
        return

    X_combined = np.concatenate(X)  # Form: (Gesamt_Frames, 42)
    
    dataset = {
        "X": X_combined,
        "lengths": lengths,
        "labels":  labels,
        "classes": classes,
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset erfolgreich gespeichert → {output_path}")
    print(f"{len(classes)} Klassen · {len(lengths)} Sequenzen · {len(X_combined)} Frames gesamt")
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser("labeling")
    parser.add_argument("--single",      action="store_true", default=False)
    parser.add_argument("--save-path",   type=str)
    parser.add_argument("--finger-idx",  type=int, default=8)
    args, _ = parser.parse_known_args()
 
    if args.single:
        
        _single_recording(
            save_path  = Path(args.save_path),
            finger_idx = args.finger_idx,
        )
    else:
        
        GESTEN    = ["P"]     # Gesten anpassen WP
        AUFNAHMEN = 20      # Anzahl Aufnahmen pro Geste
 
        for geste in GESTEN:
            data_labeling(times=AUFNAHMEN, label=geste)
 
        dataset_building("data/dataset.pickle")