import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
# Stelle sicher, dass feature_engineering im selben Ordner existiert oder passe den Import an
try:
    from feature_engineering import _extract_features
except ImportError:
    # Fallback-Dummy, falls das Modul beim Testen fehlt
    def _extract_features(seq): return np.hstack([seq, np.zeros((len(seq), 88))])

TARGET_FRAMES = 65

def _normalize(pts: np.ndarray) -> np.ndarray:
    """Zentrieren + Frame-weise Skalieren → (T, 63)"""
    T = pts.shape[0]
    lm = pts.reshape(T, 21, 3)
    wrist = lm[:, 0:1, :]
    lm = lm - wrist
    for t in range(T):
        d = np.linalg.norm(lm[t], axis=1).max()
        if d > 1e-6:
            lm[t] /= d
    return lm.reshape(T, 63)

def _resample(traj: np.ndarray, target_frames: int = 65) -> np.ndarray:
    """Interpoliert (T, 63) → (target_frames, 63)"""
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

def visualize_dataset():
    """Visualisiert alle extrahierten Fingerspitzen-Trajektorien in einem Grid."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ Datenordner 'data' existiert nicht.")
        return
        
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    seqs_by_label = {}
    for label in classes:
        seqs = []
        for npy in sorted((data_dir / label).glob("*.npy")):
            pts = np.load(npy)               # (T, 21, 3)
            seq = pts.reshape(len(pts), -1)  # (T, 63)
            
            # Modell-Pipeline anwenden
            seq = _normalize(seq)              
            seq = _resample(seq, TARGET_FRAMES) 
            seq = _extract_features(seq)       # (65, 151)
            
            # Aus dem flachen Array die normalisierten x,y-Werte von Landmark 8 holen
            tip_normalized = seq[:, 24:26] 
            seqs.append(tip_normalized)
        seqs_by_label[label] = seqs

    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    axes = axes.flatten()

    for i, label in enumerate(classes):
        ax = axes[i]
        seqs = seqs_by_label.get(label, [])

        for tip in seqs:
            ax.plot(tip[:, 0], tip[:, 1], linewidth=0.8, alpha=0.5)

        ax.set_title(f"{label} ({len(seqs)})", fontsize=10)
        ax.set_xlim(-1.0, 1.0)   # Angepasst an die echten 3D-Skalierungswerte
        ax.set_ylim(1.0, -1.0)   # Invertiert für Kamera
        ax.set_aspect("equal")
        ax.set_xlabel("x'", fontsize=7)
        ax.set_ylabel("y'", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Vollständig normalisierte Trajektorien pro Klasse", fontsize=14)
    plt.tight_layout()
    plt.show()

def evaluate_classifier():
    """TODO: Evaluation des Classifiers implementieren."""
    pass

def replay_recordings():
    """Interaktiver Explorer für deine *.npy Rohdaten mit echter Modell-Normalisierung."""
    data_dir = Path("data")
    all_files = sorted(list(data_dir.glob("*/*.npy")))
    
    if not all_files:
        print("✗ Keine .npy Aufnahmen in data/ gefunden.")
        return

    state = {"current_idx": 0}

    fig, (ax_hand, ax_traj) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    # Diese Referenzen müssen außerhalb von update_plot leben, damit die Buttons nicht sterben
    buttons = {}

    def update_plot():
        ax_hand.clear()
        ax_traj.clear()
        
        file_path = all_files[state["current_idx"]]
        label = file_path.parent.name
        pts = np.load(file_path)  # (T, 21, 3)
        
        # --- 1. RECHTS: Deine perfekte Bounding-Box-Normalisierung für die HMM ---
        seq_2d = pts.reshape(len(pts), -1)
        seq_norm = _normalize(seq_2d)  # Nutzt deine neue Bounding-Box-Funktion!
        seq_resampled = _resample(seq_norm, TARGET_FRAMES)
        seq_features = _extract_features(seq_resampled)
        
        # Flugbahn direkt aus den neu normierten Features holen
        tip = seq_resampled.reshape(TARGET_FRAMES, 21, 3)[:, 8, :2]

        # --- 2. LINKS: Lokale Handgelenk-Normalisierung NUR FÜR DAS REPLAY-AUGE ---
        # Das sorgt dafür, dass das Handmodell links groß und scharf gezeichnet wird
        hand_pts_3d = pts.copy()
        hand_wrist = hand_pts_3d[:, 0:1, :]
        hand_norm = hand_pts_3d - hand_wrist
        # Hand auf Eigen-Spannweite skalieren, damit sie groß im Plot steht
        hand_scale = np.linalg.norm(hand_norm, axis=2).max()
        if hand_scale > 1e-6:
            hand_norm /= hand_scale
            
        mid_frame = len(pts) // 2
        hand = hand_norm[mid_frame]  # Nutzt den echten mittleren Roh-Frame

        # --- PLOT 1: Trajektorie des Zeigefingers (Befüllt deine Box [0, 1]) ---
        ax_traj.plot(tip[:, 0], tip[:, 1], color="blue", label="Trajektorie (Index 8)", zorder=1)
        ax_traj.scatter(tip[0, 0], tip[0, 1], color="green", s=100, label="Start", zorder=2)
        ax_traj.scatter(tip[-1, 0], tip[-1, 1], color="red", s=100, label="Ende", zorder=2)
        
        ax_traj.set_title(f"Zeigefinger-Trajektorie (Resampled)")
        ax_traj.set_xlim(-0.1, 1.1)  # Bereich leicht erweitert, da deine Box von 0 bis 1 geht
        ax_traj.set_ylim(1.1, -0.1)  
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(loc="upper right", fontsize=8)

        # --- PLOT 2: Handform (Jetzt wieder wunderschön zentriert und groß) ---
        connections = [
            (0,1), (1,2), (2,3), (3,4),       # Daumen
            (0,5), (5,6), (6,7), (7,8),       # Zeigefinger
            (0,9), (9,10), (10,11), (11,12),  # Mittelfinger
            (0,13), (13,14), (14,15), (15,16),# Ringfinger
            (0,17), (17,18), (18,19), (19,20),# Kleiner Finger
            (5,9), (9,13), (13,17)            # Handteller
        ]

        for connection in connections:
            p1, p2 = hand[connection[0]], hand[connection[1]]
            ax_hand.plot([p1[0], p2[0]], [p1[1], p2[1]], color="purple", alpha=0.6)

        ax_hand.scatter(hand[:, 0], hand[:, 1], color="red", s=25, zorder=3)
        ax_hand.scatter(hand[0, 0], hand[0, 1], color="green", s=70, zorder=4)
        
        ax_hand.set_title(f"Lokale Handform (Frame {mid_frame})")
        ax_hand.set_xlim(-1.0, 1.0)
        ax_hand.set_ylim(1.0, -1.0)
        ax_hand.set_aspect("equal")
        ax_hand.grid(True, alpha=0.3)

        fig.suptitle(f"Replay: Klasse '{label}' | Datei: {file_path.name} ({state['current_idx']+1}/{len(all_files)})", fontsize=12)
        plt.draw()
    # Interaktive Buttons initialisieren
    from matplotlib.widgets import Button
    ax_prev = plt.axes([0.3, 0.05, 0.15, 0.075])
    ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])
    
    buttons['prev'] = Button(ax_prev, '← Zurück')
    buttons['next'] = Button(ax_next, 'Weiter →')

    def next_action(event):
        state["current_idx"] = (state["current_idx"] + 1) % len(all_files)
        update_plot()

    def prev_action(event):
        state["current_idx"] = (state["current_idx"] - 1) % len(all_files)
        update_plot()

    buttons['next'].on_clicked(next_action)
    buttons['prev'].on_clicked(prev_action)

    update_plot()
    plt.show()

if __name__ == "__main__":
    # Schalte hier um, um zwischen Grid-Ansicht (1) und interaktivem Replay (3) zu wechseln
    modus = 3
    
    if modus == 1:
        visualize_dataset()
    elif modus == 3:
        replay_recordings()