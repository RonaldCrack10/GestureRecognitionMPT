import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def visualize_class(label: str):
    data_dir = Path("data") / label
    npy_files = sorted(data_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"Keine Dateien für '{label}' gefunden.")
        return

    current = [0]
    to_delete = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Klasse: {label}  |  Navigation: ← →  |  D = löschen markieren", fontsize=13)

    ax_hand  = axes[0]
    ax_traj  = axes[1]

    ax_prev   = plt.axes([0.05, 0.02, 0.12, 0.06])
    ax_next   = plt.axes([0.19, 0.02, 0.12, 0.06])
    ax_delete = plt.axes([0.44, 0.02, 0.15, 0.06])
    ax_save   = plt.axes([0.80, 0.02, 0.15, 0.06])

    btn_prev   = Button(ax_prev,   '← Zurück')
    btn_next   = Button(ax_next,   'Weiter →')
    btn_delete = Button(ax_delete, 'Löschen markieren', color='salmon')
    btn_save   = Button(ax_save,   'Änderungen speichern', color='lightgreen')

    def draw(idx):
        ax_hand.clear()
        ax_traj.clear()

        npy = npy_files[idx]
        pts = np.load(npy)          # (T, 21, 3)
        marked = npy in to_delete

        color = 'red' if marked else 'black'
        status = '🗑 MARKIERT ZUM LÖSCHEN' if marked else ''

        # Mittlerer Frame für Handform
        mid = len(pts) // 2
        hand = pts[mid]             # (21, 3)

        ax_hand.set_title(f"Handform (Frame {mid})  {status}", color=color)
        ax_hand.set_xlim(0, 1)
        ax_hand.set_ylim(1, 0)
        ax_hand.set_aspect('equal')

        for a, b in CONNECTIONS:
            ax_hand.plot([hand[a,0], hand[b,0]],
                        [hand[a,1], hand[b,1]],
                        'b-', linewidth=1.5, alpha=0.6)
        ax_hand.scatter(hand[:,0], hand[:,1], c='red', s=30, zorder=5)
        ax_hand.scatter(hand[0,0], hand[0,1], c='green', s=80, zorder=6)

        # Trajektorie der Zeigefingerspitze
        traj = pts[:, 8, :]         # Landmark 8
        ax_traj.set_title(f"Trajektorie Zeigefinger  ({len(pts)} Frames)")
        ax_traj.plot(traj[:,0], traj[:,1], 'b-', linewidth=1.5)
        ax_traj.scatter(traj[0,0], traj[0,1], c='green', s=80, zorder=5, label='Start')
        ax_traj.scatter(traj[-1,0], traj[-1,1], c='red', s=80, zorder=5, label='Ende')
        ax_traj.set_xlim(0, 1)
        ax_traj.set_ylim(1, 0)
        ax_traj.set_aspect('equal')
        ax_traj.legend()

        fig.suptitle(
            f"Klasse: {label}  |  {idx+1}/{len(npy_files)}  |  {npy.name}  |  "
            f"{len(to_delete)} markiert",
            fontsize=12, color=color if marked else 'black'
        )
        fig.canvas.draw_idle()

    def on_prev(event):
        current[0] = (current[0] - 1) % len(npy_files)
        draw(current[0])

    def on_next(event):
        current[0] = (current[0] + 1) % len(npy_files)
        draw(current[0])

    def on_delete(event):
        npy = npy_files[current[0]]
        if npy in to_delete:
            to_delete.remove(npy)
            print(f"  ↩ Markierung aufgehoben: {npy.name}")
        else:
            to_delete.append(npy)
            print(f"  🗑 Markiert: {npy.name}")
        draw(current[0])

    def on_save(event):
        if not to_delete:
            print("Nichts zu löschen.")
            return
        print(f"\n{len(to_delete)} Dateien werden gelöscht:")
        for f in to_delete:
            print(f"  ✗ {f.name}")
            f.unlink()
        print("Fertig — bitte dataset_building neu ausführen.")
        plt.close()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_delete.on_clicked(on_delete)
    btn_save.on_clicked(on_save)

    draw(0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


# Nutzung — Label anpassen:
visualize_class("A")