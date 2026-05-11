import pickle
import numpy as np
# Ersetze den Pfad mit einer eurer heruntergeladenen .pkl Dateien
dateipfad = r"C:\Users\Evran\GestureRecognitionMPT\recordings\A\A-1773050612.172112.pkl"

with open(dateipfad, "rb") as f:
    replay_daten = pickle.load(f)

recorded_moduls= replay_daten.keys()
print("Aufgezeichnete Module:", recorded_moduls)

#drittes_modul = list(replay_daten.keys())[0]
#print(f"\nErster Frame von Modul '{drittes_modul}':")
preprocessor_frames= replay_daten.get("preprocessor")
#preprocessor_frames_type= [type(x) for x in preprocessor_frames ]
preprocessor_filled_frames = [
    x for x in preprocessor_frames
    if x is not None and x.get("preprocessor") is not None
]
#print(preprocessor_frames_type)
print(f" preprocessor frames ({len(preprocessor_frames)}):", preprocessor_frames)
#print(f"Preprocessor filled Frames ({len(preprocessor_filled_frames)}):", preprocessor_filled_frames)

#hidden_markov_data= replay_daten.get("hiddenmarkov")
#print(hidden_markov_data)