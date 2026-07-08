import numpy as np
from pathlib import Path

def clean_all_npy_files():
    data_dir = Path("data")
    MIN_FRAMES = 15
    
    print("Starte einmalige Datensatz-Bereinigung...\n")
    
    # Gehe durch alle Unterordner (A-Z)
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name == "dataset.pickle":
            continue
            
        print(f"Bereinige Klasse: {label_dir.name}...")
        
        for npy_path in sorted(label_dir.glob("*.npy")):
            try:
                pts = np.load(npy_path)
                
                # 1. Dimensionen vereinheitlichen
                if pts.ndim == 2:
                    if pts.shape[1] == 63:
                        pts = pts.reshape(-1, 21, 3)
                    elif pts.shape[1] == 42:
                        pts = pts.reshape(-1, 21, 2)
                    else:
                        print(f"  ✗ {npy_path.name}: Falsche Spaltenanzahl ({pts.shape[1]}). Lösche Datei.")
                        npy_path.unlink()
                        continue

                if pts.ndim == 3:
                    pts = pts[:, :, :2]  # Z-Achse abschneiden
                else:
                    print(f"  ✗ {npy_path.name}: Ungültige Dimensionen {pts.shape}. Lösche Datei.")
                    npy_path.unlink()
                    continue

                # 2. Flachklopfen auf (T, 42)
                pts_flat = pts.reshape(len(pts), 42)

                # 3. Zu kurze Sequenzen direkt aussortieren
                if len(pts_flat) < MIN_FRAMES:
                    print(f"  ✗ {npy_path.name}: Zu kurz ({len(pts_flat)} Frames). Lösche Datei.")
                    npy_path.unlink()
                    continue
                
                # 4. Die sauber formatierte Datei direkt überschreiben!
                np.save(npy_path, pts_flat)
                
            except Exception as e:
                print(f"  ✗ Fehler bei {npy_path.name}: {e}. Lösche Datei.")
                if npy_path.exists():
                    npy_path.unlink()

    print("\nBereinigung abgeschlossen! Alle verbleibenden .npy-Dateien haben jetzt exakt das Format (T, 42).")

if __name__ == "__main__":
    clean_all_npy_files()