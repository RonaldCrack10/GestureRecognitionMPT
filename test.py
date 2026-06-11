# import numpy as np
# pts = np.load("data/A/A_011.npy")
# print(pts.shape)
# print(pts[0])

from collections import Counter
import pickle

with open("data/dataset.pickle", "rb") as f:
    ds = pickle.load(f)

print(ds["X"].shape)
print(Counter(ds["labels"]))
