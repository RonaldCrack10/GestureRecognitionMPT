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

    def _is_fist(self, landmarks, threshold: float = 0.15) -> bool:
        """Gibt True zurück wenn die Hand zur Faust geschlossen ist."""
        FINGERTIPS = [8, 12, 16, 20]
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        for tip_idx in FINGERTIPS:
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
            if np.linalg.norm(tip - wrist) > threshold:
                return False
        return True

   
    def _resample(self, traj: np.ndarray) -> np.ndarray:
        """Interpoliert (T, 63) → (target_frames, 63)"""
        T = traj.shape[0]
        if T == self.target_frames:
            return traj
        f = interp1d(
            np.linspace(0, 1, T),
            traj,
            axis=0,
            kind='linear',
            fill_value="extrapolate"
        )
        return f(np.linspace(0, 1, self.target_frames)).astype(np.float32)

    def start(self, data):
       
        config = data.get("config", {}).get("preprocessor", {})
        self.buffer_size   = config.get("buffer_size",   140)
        self.max_lost      = config.get("max_lost",       10)
        self.min_steps     = config.get("min_steps",      15)
        self.target_frames = config.get("target_frames",  65)
        self.history       = deque(maxlen=self.buffer_size)
        self.lost_frames   = 0
        return {}

    
    def step(self, data):
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
                res  = self._resample(norm)     
                result_trajectory = res        
            

        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                if len(self.history) >= self.min_steps:
                    raw = np.array(self.history, dtype=np.float32)
                    norm = _normalize_trajectory_only(raw)
                    res  = self._resample(norm)     
                    result_trajectory = res

                    
                self.history.clear()
                self.lost_frames = 0

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        self.history.clear()