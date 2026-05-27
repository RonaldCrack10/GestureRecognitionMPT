import msvcrt
import numpy as np
from pathlib import Path
from SignalHub import Module


class _StopRecording(Exception):
    pass


class DataRecorder(Module):
    """
    SignalHub-Modul zur Aufnahme von Handgesten-Landmarken.

    Subscribt auf das 'detector'-Signal des HandDetectors,
    sammelt Fingerpositionen pro Frame und speichert auf ESC-Druck
    als .npy-Datei. Stoppt die Engine danach automatisch.

    Parameters
    ----------
    save_path : Path
        Zielpfad für die .npy-Datei.
    finger_idx : int
        Index des Landmarks, der aufgezeichnet wird.
        Standard: 8 (Zeigefingerspitze).
    min_frames : int
        Mindestanzahl gültiger Frames, sonst wird nicht gespeichert.
    """

    def __init__(self, save_path: Path, finger_idx: int = 8, min_frames: int = 15):
        self.save_path  = Path(save_path)
        self.finger_idx = finger_idx
        self.min_frames = min_frames
        self.points: list = []
        self.saved  = False

        super().__init__(
            name         = "DataRecorder",
            inputSignals = ["detector"],
            outputSchema = {}
        )

    # ------------------------------------------------------------------
    def start(self, data: dict) -> dict:
        self.points = []
        self.saved  = False

        # Tastenpuffer leeren
        while msvcrt.kbhit():
            msvcrt.getch()

        print("\n  ● Aufnahme läuft — ESC zum Speichern drücken")
        return {}

    # ------------------------------------------------------------------
    def step(self, data: dict) -> dict:
        # Landmark extrahieren
        det = data.get("detector")
        if det is not None and det.hand_landmarks:
            lm = det.hand_landmarks[0][self.finger_idx]
            self.points.append([lm.x, lm.y])

        # ESC-Check (non-blocking)
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"\x1b":          # ESC
                self._finalize()
                raise _StopRecording()     # Engine sauber stoppen

        return {}

    # ------------------------------------------------------------------
    def stop(self, data: dict) -> None:
        # Wird auch beim normalen Engine-Stop aufgerufen
        if not self.saved:
            self._finalize()

    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        """Speichert Daten oder gibt Warnung aus."""
        n = len(self.points)
        if n < self.min_frames:
            print(f"\n  ✗ Zu wenig Frames ({n}/{self.min_frames}) — nicht gespeichert.")
        else:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            arr = np.array(self.points, dtype=np.float32)
            np.save(self.save_path, arr)
            self.saved = True
            print(f"\n  ✓ Gespeichert: {self.save_path}  ({n} Frames)")