import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAW_DATA_DIR = Path("data/raw")

def get_key():
    try:
        import msvcrt
        return msvcrt.getch().decode(errors="ignore")
    except ImportError:
        import tty
        import termios

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

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
    label_dir = RAW_DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Labeling für Klasse: {label}")
    print("ESC = speichern | andere Taste = verwerfen | q = beenden")

    saved = 0

    while saved < times:
        timestamp = int(time.time() * 1000)
        temp_path = label_dir / f"_temp_{timestamp}.pkl"
        final_path = label_dir / f"{label}_{timestamp}.pkl"

        process = subprocess.Popen(
            [
                "SignalHub",
                "--mode",
                "record",
                "--recorder",
                str(temp_path),
            ]
        )

        print(f"\nAufnahme {saved + 1}/{times}")
        print("Geste ausführen, dann Taste drücken.")

        key = get_key()

        process.terminate()
        process.wait()

        if key == "q":
            if temp_path.exists():
                temp_path.unlink()
            print("Beendet.")
            break

        if key == "\x1b":
            if temp_path.exists():
                shutil.move(str(temp_path), str(final_path))
                saved += 1
                print(f"Gespeichert: {final_path}")
            else:
                print("Keine Aufnahme gefunden.")
        else:
            if temp_path.exists():
                temp_path.unlink()
            print("Verworfen.")



def load_recording(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
    
def preprocess_sequence(recording):
    """
    Erwartete Recording-Struktur:

    recording["detector"] = Liste von Frames
    """
    try:
        frames = recording["preprocessor"]
    except (KeyError, TypeError):
        return None

    sequence = []

    for frame in frames:
        if frame is None:
            continue

        arr = np.asarray(frame, dtype=np.float32)

        if arr.size == 0:
            continue

        sequence.append(arr)

    if len(sequence) == 0:
        return None
    return sequence[-1]

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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_DIR.exists():
        print(f"Trainingsordner existiert nicht: {RAW_DATA_DIR}")
        return None

    X = []
    y = []
    lengths = []

    for label_dir in RAW_DATA_DIR.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        for file_path in label_dir.glob("*.pkl"):
            if file_path.name.startswith("_temp_"):
                continue

            recording = load_recording(file_path)
            sequence = preprocess_sequence(recording)

            if sequence is None:
                print(f"Übersprungen: {file_path}")
                continue

            if len(sequence) < 5:
                print(f"Zu kurz: {file_path}")
                continue

            X.append(sequence)
            y.append(label)
            lengths.append(len(sequence))

    dataset = {
        "X": X,
        "y": y,
        "lengths": lengths,
        "labels": sorted(set(y)),
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset gespeichert unter: {output_path}")
    print(f"Sequenzen: {len(X)}")
    print(f"Labels: {dataset['labels']}")

    return dataset