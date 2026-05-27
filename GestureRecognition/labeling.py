import argparse
import numpy as np
import msvcrt
from pathlib import Path

from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules.handdetector import HandDetector
from GestureRecognition.modules.datarecorder import DataRecorder
from GestureRecognition.modules.trailmarker import TrailMarker




def _normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    d   = np.linalg.norm(pts, axis=1).max()
    if d > 1e-6:
        pts /= d
    return pts




def data_labeling(times: int, label: str, finger_idx: int = 8):
    """
    Startet die HandDetector-Pipeline für ``times`` Aufnahmen der Geste
    ``label``. ESC beendet jede Aufnahme und speichert sie als .npy.

    Parameters
    ----------
    times : int
        Anzahl der gewünschten Aufnahmen.
    label : str
        Name der Geste (wird als Unterordner unter data/raw/ verwendet).
    finger_idx : int
        MediaPipe-Landmark-Index (Standard 8 = Zeigefingerspitze).
    """
    save_dir = Path("data") / "raw" / label
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    while saved < times:
        save_path = save_dir / f"{label}_{saved + 1:03d}.npy"

        print(f"\n[{label}]  Aufnahme {saved + 1}/{times}")
        print(f"  Datei: {save_path}")

        recorder = DataRecorder(save_path=save_path, finger_idx=finger_idx)

        # Frischer Parser pro Durchlauf (Engine erwartet unverarbeiteten Parser)
        parser = argparse.ArgumentParser("GestureRecognition")
        parser.add_argument("--mode", action="store", default="none")

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
        except SystemExit:
            pass  # sauberer ESC-Stop

        # Prüfen ob tatsächlich gespeichert wurde
        if recorder.saved:
            saved += 1
        else:
            print("  → Nochmal versuchen? (j / andere Taste = abbrechen)")
            # Tastenpuffer leeren
            while msvcrt.kbhit():
                msvcrt.getch()
            key = msvcrt.getch().lower()
            if key != b"j":
                print("  Abgebrochen.")
                break

    print(f"\nFertig: {saved}/{times} Aufnahmen für '{label}' gespeichert.")




def dataset_building(output_path):
    """
    Lädt alle .npy-Aufnahmen, normalisiert sie und speichert einen
    hmmlearn-kompatiblen Datensatz als pickle.

    Parameters
    ----------
    output_path : str or Path
        Zielpfad für die dataset.pickle-Datei.
    """
    import pickle

    raw_dir     = Path("data") / "raw"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MIN_FRAMES = 15
    X, lengths, labels, classes = [], [], [], []

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        classes.append(label)
        count = 0

        for npy in sorted(label_dir.glob("*.npy")):
            pts = np.load(npy)

            if len(pts) < MIN_FRAMES:
                print(f"  ✗ {label}/{npy.name}: {len(pts)} Frames — übersprungen")
                continue

            seq = _normalize(pts)
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
        "X":       np.concatenate(X),
        "lengths": lengths,
        "labels":  labels,
        "classes": classes,
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset gespeichert → {output_path}")
    print(f"{len(classes)} Klassen · {len(lengths)} Sequenzen · {len(dataset['X'])} Punkte gesamt")




if __name__ == "__main__":
    GESTEN    = ["A"]   # ← Gesten anpassen
    AUFNAHMEN = 2            # ← Anzahl Aufnahmen pro Geste

    for geste in GESTEN:
        data_labeling(times=AUFNAHMEN, label=geste)

    dataset_building("data/dataset.pickle")