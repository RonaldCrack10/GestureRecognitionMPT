import msvcrt

from SignalHub import GALY, Module
from collections import deque
import numpy as np
from scipy.interpolate import interp1d
from GestureRecognition.labeling import _normalize_trajectory_only


class Preprocessor(Module):

    def __init__(self, outputSignal="preprocessor"):
        self.outputSignal = outputSignal
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )


    def start(self, data):
       
        config = data.get("config", {}).get("preprocessor", {})
        self.buffer_size = config.get("buffer_size", 140) # maximale Anzahl an Frames, die im Speicher gehalten werden
        self.max_lost = config.get("max_lost", 10) # maximale Anzahl an aufeinanderfolgenden Frames, in denen keine Hand erkannt wird, bevor die Sequenz als beendet betrachtet wird
        self.min_steps = config.get("min_steps", 15) # minimale Anzahl an Frames, die für eine gültige Sequenz benötigt werden
        self.target_frames = config.get("target_frames", 70) 
        self.history = deque(maxlen=self.buffer_size) # Speichert die letzten N Frames der Handlandmarken
        self.lost_frames = 0
        self.paused = False  
        return {}

    def _resample(self, traj: np.ndarray, target_frames: int = 70) -> np.ndarray: # diese Funktion macht die Interpolation der Trajektorie auf eine feste Länge (0, 1)
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

    
    def step(self, data):

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"a" : 
                self.paused = not self.paused
                if self.paused:
                    print("Detektion pausiert. Drücke Leertaste zum Fortsetzen.")
                    self.history.clear()  # Leere den Verlauf, wenn die Detektion pausiert wird
                else:
                    print("Detektion fortgesetzt.")

        if self.paused:
            return {self.outputSignal: None}


        result = data.get("detector")
        result_trajectory = None

        if result is not None and result.hand_landmarks:
            self.lost_frames = 0
            landmarks = result.hand_landmarks[0]
            frame = [coord for lm in landmarks for coord in (lm.x, lm.y)]
            self.history.append(frame)

            if len(self.history) >= self.min_steps:
                raw = np.array(self.history, dtype=np.float32)
                norm = _normalize_trajectory_only(raw)
                # res  = self._resample(norm)     
                result_trajectory = norm        
            

        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                if len(self.history) >= self.min_steps:
                    raw = np.array(self.history, dtype=np.float32)
                    norm = _normalize_trajectory_only(raw)
                    #res  = self._resample(norm)     
                    result_trajectory = norm

                    
                self.history.clear()
                self.lost_frames = 0

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        self.history.clear()