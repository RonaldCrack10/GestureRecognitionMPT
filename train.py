import pickle
import numpy as np
from GestureRecognition.hmmclassifier import HMMClassifier
from collections import Counter

with open("data/dataset.pickle", "rb") as f:
    ds = pickle.load(f)

# Manueller Split vor fit()
rng = np.random.default_rng(42)
lengths = ds["lengths"]
labels = ds["labels"]
n = len(lengths)

perm = rng.permutation(n)
n_test = int(n * 0.2)
train_idx = perm[n_test:]
test_idx = perm[:n_test]

# Sequenzen rekonstruieren
seqs = []
idx = 0
for length in lengths:
    seqs.append(ds["X"][idx:idx+length])
    idx += length

train_seqs = [seqs[i] for i in train_idx]
test_seqs  = [seqs[i] for i in test_idx]
train_labels = [labels[i] for i in train_idx]
test_labels  = [labels[i] for i in test_idx]

X_train = np.concatenate(train_seqs)
X_test  = np.concatenate(test_seqs)
lens_train = [len(s) for s in train_seqs]
lens_test  = [len(s) for s in test_seqs]


clf = HMMClassifier(n_components=8, n_iter=200, test_size=0.0)  # kein interner split
clf.fit(X_train, lens_train, train_labels)
clf.evaluate(X_test, lens_test, test_labels)
clf.save("data/hmm_model.pickle")