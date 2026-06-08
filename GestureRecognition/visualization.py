import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.misc.classifyTools import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix
from hmmclassifier import HMMClassifier

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
#if __name__ == "__main__":
    #test_ordner_O = r"C:\Users\Evran\GestureRecognitionMPT\recordings\O"
    #visualize_dataset(test_ordner_O, "Buchstabe O")

def evaluate_classifier(model_path, test_data_path):
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
    print("Starts evaluation...")
    print(f"lade modell aus {model_path}")
    print(f"lade Test daten aus {test_data_path}")

    classifier = HMMClassifier()
    try:
        classifier.load_model(model_path)
        classifier.load_test_data(test_data_path)
    except FileNotFoundError:
        print("Fehler: Konnte Modell oder Testdaten nicht finden.")
        return

    if not classifier.test_data:
        print("Fehler: Keine Testdaten geladen!")
        return

    y_true = []
    y_pred = []

    print("Predictions start....")

    for true_label, sequences_list in classifier.test_data.items():
        if not sequences_list:
            continue

        predictions = classifier.predict(sequences_list)
        y_true.extend([true_label] * len(sequences_list))
        y_pred.extend(predictions)

    # Metriken
    genauigkeit= accuracy_score(y_true, y_pred)
    print(f"Klassifikationsgenauigkeit (Accuracy): {genauigkeit:.2%} ({genauigkeit * 100:.1f}%)")

    # 5. Confusion Matrix zeichnen
    alle_klassen = sorted(list(set(y_true + y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=alle_klassen)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=alle_klassen, yticklabels=alle_klassen)

    plt.title(f'Confusion Matrix\n(Genauigkeit: {genauigkeit:.2%})', fontsize=16)
    plt.ylabel('Wahre Geste', fontsize=12)
    plt.xlabel('Vorausgesagte Geste', fontsize=12)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mein_modell = "trained_models/hmm_models.pkl"
    meine_testdaten = "test_data/test_data.pkl"

    evaluate_classifier(mein_modell, meine_testdaten)


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