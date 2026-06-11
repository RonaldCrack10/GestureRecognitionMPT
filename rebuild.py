# from GestureRecognition.labeling import dataset_building
# dataset_building("data/dataset.pickle")
# rebuild.py
import importlib
import GestureRecognition.labeling as lab
importlib.reload(lab)
lab.dataset_building("data/dataset.pickle")