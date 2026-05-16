from SignalHub import GALY, Module
from collections import deque
import numpy as np


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

        self.finger_idx  = config.get("finger_idx",  8)
        self.buffer_size = config.get("buffer_size", 140)
        self.max_lost    = config.get("max_lost",    10)
        self.min_steps   = config.get("min_steps",   15)

        self.history     = deque(maxlen=self.buffer_size)
        self.lost_frames = 0
        return {}

    def step(self, data):
        result           = data.get("detector")
        result_trajectory = None

        if result is not None and result.hand_landmarks:
            self.lost_frames = 0

            # korrekte Struktur: result.hand_landmarks[0][idx]
            lm = result.hand_landmarks[0][self.finger_idx]
            self.history.append([lm.x, lm.y])

        else:
            self.lost_frames += 1

        # Geste beendet wenn Hand lange genug weg
        if self.lost_frames > self.max_lost:
            if len(self.history) >= self.min_steps:
                traj = np.array(self.history, dtype=np.float32)

                # Zentrieren
                traj -= traj.mean(axis=0)

                # Skalieren nach weitestem Punkt
                max_dist = np.linalg.norm(traj, axis=1).max()
                if max_dist > 1e-6:
                    traj /= max_dist

                result_trajectory = traj
                # print(f"✅ Geste erfasst! {len(traj)} Punkte")

            self.history.clear()

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        self.history.clear()