import pickle
from GestureRecognition.hmmclassifier import HMMClassifier

with open("data/dataset.pickle", "rb") as f:
    ds = pickle.load(f)

clf = HMMClassifier(n_components=6, n_iter=300, test_size=0.2)
clf.fit(ds["X"], ds["lengths"], ds["labels"])
clf.evaluate(ds["X"], ds["lengths"], ds["labels"])
clf.save("data/hmm_model.pickle")