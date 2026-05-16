import sys
import os
import pickle
import subprocess
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
    def getch():
        return msvcrt.getch()
else:
    def getch():
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1).encode()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clean(raw: list) -> list | None:
    """
    Entfernt Frames ohne erkannte Hand am Anfang und Ende.
    Gibt None zurück wenn weniger als 15 brauchbare Frames übrig bleiben.
    """
    MIN_FRAMES = 15

    # Nur Frames behalten wo preprocessor-Daten vorhanden sind
    frames = [f for f in raw if f.get("preprocessor") is not None]

    return frames if len(frames) >= MIN_FRAMES else None

def data_labeling(times: int, label: str):
    """
    TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

    Ziel:
    -----
    Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
    Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess
       - Übergib einen Dateipfad für die Aufnahme
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    2. Interaktive Steuerung (optional)

       - Implementiere eine einfache Benutzerinteraktion:
         - Aufnahme speichern
         - Aufnahme verwerfen
         - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

           ESC → speichern
           andere Taste → verwerfen

    3. Daten sichten und bereinigen

       - Lade die aufgenommenen Daten
       - Überlege:
         - Welche Teile sind relevant?
         - Welche Frames sind leer oder unbrauchbar?
         - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
       - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gemacht werden sollen.
        Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).

    label : str
        Name der Geste / Klasse.
        Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).
    """
    save_dir = Path("data") / "raw" / label
    save_dir.mkdir(parents=True, exist_ok=True)

    saved   = 0
    attempt = 0

    while saved < times:
        attempt += 1
        tmp_path = save_dir / f"tmp_{attempt}.pickle"

        print(f"\n[{label}] Aufnahme {saved+1}/{times}")
        print("  Geste ausführen → ESC = speichern | andere Taste = verwerfen | q = beenden")

        # SignalHub Demo als Subprocess starten mit Recorder
        proc = subprocess.Popen([
            sys.executable,
            "GestureRecognition/demo.py",
            "--recorder.file", str(tmp_path),
        ])

        key = getch()          # blockiert bis Taste gedrückt
        proc.terminate()
        proc.wait()

        # q → komplett abbrechen
        if key in (b'q', b'Q'):
            if tmp_path.exists():
                tmp_path.unlink()
            print("Abgebrochen.")
            break

        # ESC → speichern
        if key == b'\x1b':
            if not tmp_path.exists():
                print("  ✗ Keine Datei erzeugt — verworfen.")
                continue

            with open(tmp_path, "rb") as f:
                raw = pickle.load(f)

            cleaned = _clean(raw)
            tmp_path.unlink()

            if cleaned is None:
                print("  ✗ Zu wenig brauchbare Frames — verworfen.")
                continue

            out = save_dir / f"{label}_{saved+1:03d}.pickle"
            with open(out, "wb") as f:
                pickle.dump(cleaned, f)

            saved += 1
            print(f"  ✓ Gespeichert: {out}  ({len(cleaned)} Frames)")

        # andere Taste → verwerfen
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            print("  ✗ Verworfen.")

    print(f"\nFertig: {saved}/{times} Aufnahmen für '{label}'.")




def dataset_building(output_path):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """
    raw_dir     = Path("data") / "raw"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MIN_FRAMES = 15
    dataset    = {}   # { label: [np.ndarray(N,2), ...] }

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label     = label_dir.name
        sequences = []

        for pkl in sorted(label_dir.glob("*.pickle")):
            with open(pkl, "rb") as f:
                frames = pickle.load(f)

            # Letzten Punkt jeder gespeicherten Trajektorie extrahieren
            # preprocessor liefert np.ndarray (N,2) — wir nehmen den
            # aktuellen Fingertipp (letzter Punkt) als Feature pro Frame
            points = []
            for frame in frames:
                traj = frame.get("preprocessor")
                if traj is not None and len(traj) > 0:
                    points.append(traj[-1])   # (x, y) normalisiert

            if len(points) < MIN_FRAMES:
                print(f"  ✗ {label}/{pkl.name}: nur {len(points)} Frames — übersprungen")
                continue

            seq = _normalize(np.array(points, dtype=np.float32))
            sequences.append(seq)
            print(f"  ✓ {label}/{pkl.name}: {len(seq)} Frames")

        if sequences:
            dataset[label] = sequences
            print(f"→ '{label}': {len(sequences)} Sequenzen\n")
        else:
            print(f"⚠ '{label}': keine brauchbaren Sequenzen\n")

    if not dataset:
        print("Dataset leer — keine Daten gefunden.")
        return

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    total = sum(len(v) for v in dataset.values())
    print(f"Dataset gespeichert → {output_path}")
    print(f"{len(dataset)} Klassen, {total} Sequenzen gesamt")
    for lbl, seqs in dataset.items():
        lens = [len(s) for s in seqs]
        print(f"  {lbl}: {len(seqs)} Seq. | ø{np.mean(lens):.0f} Frames "
              f"(min {min(lens)}, max {max(lens)})")


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Zentriert und skaliert (N,2)-Array auf [-1, 1]."""
    pts = pts - pts.mean(axis=0)
    d   = np.linalg.norm(pts, axis=1).max()
    if d > 1e-6:
        pts /= d
    return pts

if __name__ == '__main__':
    data_labeling(times = 5, label = 'circle')
    dataset_building("dataset/data.pickle")