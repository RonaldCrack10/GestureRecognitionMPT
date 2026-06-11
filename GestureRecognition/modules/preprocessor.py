from SignalHub import GALY, Module
from collections import deque
import numpy as np
from scipy.interpolate import interp1d  # Standard interpolation tool
from feature_engineering import _extract_features

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
        
        # Target length for the HMM (Crucial to keep log-likelihoods stable)
        self.target_frames = config.get("target_frames", 30) 

        self.history     = deque(maxlen=self.buffer_size)
        self.lost_frames = 0
        
        return {}

    def _resample_trajectory(self, traj):
        """Linearly interpolates a trajectory of shape (T, 63) to (target_frames, 63)"""
        T = traj.shape[0]
        if T == self.target_frames:
            return traj
            
        # Create time grids
        current_time_grid = np.linspace(0, 1, T)
        target_time_grid = np.linspace(0, 1, self.target_frames)
        
        # Interpolate along the time axis (axis=0)
        f = interp1d(current_time_grid, traj, axis=0, kind='linear', fill_value="extrapolate")
        return f(target_time_grid).astype(np.float32)

    def step(self, data):
        result = data.get("detector")
        result_trajectory = None

        if result is not None and result.hand_landmarks:
            self.lost_frames = 0

            # Extract 21 landmarks with x, y, z -> 63 features
            landmarks = result.hand_landmarks[0]
            frame = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            self.history.append(frame)

        else:
            self.lost_frames += 1

        # Gesture end condition: Hand left the frame
        if self.lost_frames > self.max_lost:
            if len(self.history) >= self.min_steps:
                raw_traj = np.array(self.history, dtype=np.float32)  # Shape: (T, 63)

                # 1. Wrist-relative Normalization
                # Reshape to (T, 21, 3) to easily access coordinate dimensions
                T = raw_traj.shape[0]
                reshaped_traj = raw_traj.reshape(T, 21, 3)
                wrist = reshaped_traj[:, 0, :][:, np.newaxis, :]  # Shape: (T, 1, 3)
                normalized_traj = reshaped_traj - wrist

                # 2. Frame-by-Frame Scale Normalization
                # Scale each individual frame by its own maximum landmark distance
                # This prevents sequence length or speed from affecting spatial scale
                for t in range(T):
                    frame_dist = np.linalg.norm(normalized_traj[t], axis=1).max()
                    if frame_dist > 1e-6:
                        normalized_traj[t] /= frame_dist

                # Flatten back to (T, 63)
                traj = normalized_traj.reshape(T, 63)

                # 3. Temporal Resampling
                resampled = self._resample_trajectory(traj)  # (target_frames, 63)

                # 4. Feature Engineering (Winkel + Distanzen + Velocity)
                result_trajectory = _extract_features(resampled)  # (target_frames, 156)

            self.history.clear()
            self.lost_frames = 0

        return {self.outputSignal: result_trajectory}

    def stop(self, data):
        self.history.clear()