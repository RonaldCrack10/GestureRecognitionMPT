import pickle
from cProfile import label

import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_dataset(label_pfad, label_name):
    """
    TODO: Visualisierung des eigenen Datensatzes

    Ziel:
    -----
    Entwickle eine Möglichkeit, deinen aufgenommenen Datensatz visuell zu
    inspizieren und zu verstehen.

    Warum ist das wichtig?
    ----------------------
    - Du musst nachvollziehen können, was dein Modell eigentlich „sieht“
    - Fehler im Datensatz lassen sich visuell oft sofort erkennen
    - Qualität der Daten ist entscheidend für die Modellperformance

    Anforderungen / Ideen:
    ----------------------
    - Lade deinen Trainingsdatensatz
    - Visualisiere mehrere Sequenzen pro Klasse
    - Stelle sicher, dass:
        - unterschiedliche Gesten klar unterscheidbar sind
        - Sequenzen sinnvoll aussehen (keine Ausreißer, keine leeren Daten)

    .. tip::
       Ein einfacher Ansatz:
         - Plotte Trajektorien (z. B. x/y-Koordinaten)
         - Zeige mehrere Beispiele pro Klasse übereinander

    .. note::
       Du kannst selbst entscheiden:
         - Wie viele Sequenzen du anzeigst
         - Welche Features du visualisierst
         - Ob du interaktive Elemente einbaust

    .. tip::
       Interaktivität (z. B. Klick auf eine Sequenz) kann hilfreich sein,
       um einzelne Beispiele genauer zu untersuchen.

    Abgabe:
    -------
    - Du musst in der Lage sein, deinen Datensatz visuell zu präsentieren
    - Du solltest erklären können:
        - Wie unterscheiden sich die Klassen?
        - Gibt es problematische Beispiele?

    Erweiterung (optional):
    -----------------------
    - Mittelwerte oder typische Sequenzen pro Klasse darstellen
    - Ausreißer automatisch erkennen
    """
    print(f"Loading Data from Ordner: {label_pfad} ...")

    plt.figure(figsize=(8, 8))
    dateien_gefunden = False

    # Geht durch alle Dateien im angegebenen Ordner
    for filename in os.listdir(label_pfad):
        if not filename.endswith(".pkl"):
            continue

        dateien_gefunden = True
        pfad = os.path.join(label_pfad, filename)

        try:
            with open(pfad, "rb") as f:
                recording_data = pickle.load(f)
        except Exception as e:
            print(f"Fehler beim Laden von {filename}: {e}")
            continue

        if 'preprocessor' not in recording_data:
            continue

        complete_sequence = None   #enthält die Koordinaten aller Frames, normalisiert und skaliert
        max_length = 0

        for frame_dict in recording_data['preprocessor']:
            if frame_dict is None:
                continue
            frame_daten = frame_dict.get('preprocessor')

            if frame_daten is not None and len(frame_daten) > max_length:
                max_length = len(frame_daten)
                complete_sequence = frame_daten

        # Wenn wir eine fertige Trajektorie gefunden haben;
        if complete_sequence is not None:
            x_werte = complete_sequence[:, 0]
            y_werte = complete_sequence[:, 1]

            plt.plot(x_werte, y_werte, marker='o', markersize=2, alpha=0.5)

    if not dateien_gefunden:
        print(f"Keine .pkl Dateien im Ordner {label_pfad} gefunden!")
        return

    plt.title(f"Visualisierung Datensatz: {label_name}\n(Alle Aufnahmen übereinander)", fontsize=14)
    plt.xlabel("X-Koordinate (normalisiert)")
    plt.ylabel("Y-Koordinate (normalisiert)")

    plt.grid(True, linestyle='--', alpha=0.7)


    #plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

#Test mit Buchstabe O
if __name__ == "__main__":
    test_ordner_O = r"C:\Users\Evran\GestureRecognitionMPT\recordings\O"
    visualize_dataset(test_ordner_O, "Buchstabe O")



def evaluate_classifier():
    """
    TODO: Evaluation deines Klassifikators

    Ziel:
    -----
    Implementiere eine sinnvolle Auswertung deines Modells auf Testdaten.

    Warum ist das wichtig?
    ----------------------
    - Du brauchst objektive Metriken für die Qualität deines Modells
    - Training allein reicht nicht, entscheidend ist die Generalisierung

    Anforderungen / Ideen:
    ----------------------
    - Lade ein trainiertes Modell
    - Lade Testdaten (getrennt vom Training!)
    - Berechne Vorhersagen
    - Vergleiche Vorhersagen mit Ground Truth

    Metriken:
    ---------
    - Klassifikationsgenauigkeit (Accuracy)
    - Confusion Matrix

    .. tip::
       Eine Confusion Matrix zeigt dir:
         - Welche Klassen gut erkannt werden
         - Wo dein Modell Fehler macht

    .. warning::
       Testdaten dürfen **nicht** aus dem Training stammen!

    Interpretation:
    ---------------
    Du solltest erklären können:
    - Welche Klassen gut funktionieren
    - Welche Klassen verwechselt werden
    - Warum das passieren könnte

    .. note::
       Schlechte Performance liegt oft an:
         - schlechten Trainingsdaten
         - zu wenigen Beispielen
         - ungeeigneten Features

    Erweiterung (optional):
    -----------------------
    - Weitere Metriken (Precision, Recall, F1)
    - Vergleich verschiedener Modelle
    """
    pass

def replay_recordings():
    """
    TODO: Exploration und Replay der aufgenommenen Rohdaten

    Ziel:
    -----
    Ermögliche es, aufgenommene Sequenzen erneut abzuspielen
    und qualitativ zu überprüfen.

    Warum ist das wichtig?
    ----------------------
    - Du kannst überprüfen, ob deine Aufnahmen korrekt sind
    - Fehler in der Datenerfassung werden früh sichtbar
    - Du entwickelst ein besseres Verständnis für deine Daten

    Anforderungen / Ideen:
    ----------------------
    - Lade gespeicherte Aufnahmen
    - Spiele diese erneut ab (z. B. über SignalHub / Replay-Modus)
    - Iteriere über verschiedene Labels und Beispiele

    .. tip::
       Besonders hilfreich:
         - Vergleiche mehrere Beispiele derselben Klasse
         - Suche nach inkonsistenten Bewegungen

    .. warning::
       Schlechte oder inkonsistente Aufnahmen führen fast immer zu
       schlechten Modellen. Überprüfe deine Daten frühzeitig!

    Abgabe:
    -------
    - Du solltest zeigen können, wie deine Daten aussehen (Replay)
    - Du solltest erklären können:
        - Welche Beispiele gut sind
        - Welche problematisch sind

    Erweiterung (optional):
    -----------------------
    - Automatisches Filtern schlechter Sequenzen
    - Kombination mit Visualisierung
    """
    pass