import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

with open("data/dataset.pickle", "rb") as f:
    ds = pickle.load(f)

classes = sorted(set(ds["labels"]))

seqs_by_label = defaultdict(list)
idx = 0
for length, label in zip(ds["lengths"], ds["labels"]):
    seqs_by_label[label].append(ds["X"][idx : idx + length])
    idx += length

fig, axes = plt.subplots(4, 7, figsize=(20, 12))
axes = axes.flatten()

for i, label in enumerate(classes):
    ax   = axes[i]
    seqs = seqs_by_label[label]

    for s in seqs:
        # Zeigefingerspitze x,y aus den ersten 63 Features (Landmark 8)
        tip = s[:, 24:26]  # x=24, y=25
        ax.plot(tip[:,0], tip[:,1], linewidth=0.8, alpha=0.4)

    ax.set_title(f"{label} ({len(seqs)})", fontsize=10)
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(1.3, -0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x'", fontsize=7)
    ax.set_ylabel("y'", fontsize=7)
    ax.tick_params(labelsize=6)

for j in range(len(classes), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Trajektorien nach Verarbeitung — was das Modell sieht", fontsize=14)
plt.tight_layout()
plt.show()