import os
import pickle
import random
import numpy as np
from hmmlearn import hmm


class HMMClassifier:
    """
    TODO: Implementiere einen HMM-basierten Klassifikator

    Ziel:
    -----
    Entwickle einen Klassifikator, der zeitliche Sequenzen mit Hilfe von
    Hidden-Markov-Modellen (HMMs) klassifiziert. Für HMMs können libraries wie
    :mod:`hmmlearn` benutzt werden

    Grundidee:
    ----------
    - Trainiere ein Modell pro Klasse
    - Bewerte neue Sequenzen anhand der Likelihood unter jedem Modell
    - Wähle die Klasse mit der höchsten Wahrscheinlichkeit

    .. note::
       Wie genau deine Modelle aussehen (z. B. Anzahl Zustände, Features,
       Initialisierung etc.) ist bewusst nicht vorgegeben.

    Wichtige Designentscheidungen:
    ------------------------------
    - Wie strukturierst du deine Trainingsdaten?
    - Wie repräsentierst du Sequenzen?
    - Wie verbindest du mehrere Sequenzen mit Labels?

    Speicherung:
    ------------
    Du solltest dir überlegen:
    - Wie speicherst du dein trainiertes Modell?
    - Wie lädst du es später wieder?
    - Welche Informationen müssen persistiert werden (z. B. Klassen, Modelle)?

    .. tip::
       ``pickle`` ist eine einfache Möglichkeit, Modelle zu speichern.
       Alternativ kannst du auch eigene Formate definieren.

    Evaluation:
    -----------
    Für sinnvolles Training solltest du unbedingt:
    - eine eigene ``train_test_split``-Logik implementieren
    - Trainings- und Testdaten sauber trennen

    .. warning::
       Wenn du Training und Test nicht trennst, sind deine Ergebnisse nicht aussagekräftig.

    Erweiterung (optional):
    -----------------------
    - Implementiere eine Grid Search für Hyperparameter
      (z. B. Anzahl Zustände, Modellstruktur)
    - Vergleiche verschiedene Modellkonfigurationen

    """
    def __init__(self, n_components=5, n_iter=100, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

        # Interner Speicher für die trainierten Modelle und Klassen
        self.models = {}
        self.classes_ = []

        # Speicher für Testdaten zur späteren Evaluation
        self.test_data = {}

    def save_model(self, filepath="trained_models/hmm_models.pkl"):
        """Speichert die trainierten Modelle und Klassen."""
       # filepath = os.path.join("trained_models","hmm_models.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"models": self.models, "classes": self.classes_}, f)

        print(f"Modell erfolgreich gespeichert unter: {filepath}")

    def load_model(self, filepath="trained_models/hmm_models.pkl"):
        """Lädt trainierte Modelle aus einer Pickle-Datei."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.models = data["models"]
            self.classes_ = data["classes"]
        return self

    def save_test_data(self, filepath="test_data/test_data.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.test_data, f)
        print(f"Testdaten erfolgreich gespeichert unter: {filepath}")

    def load_test_data(self, filepath="test_data/test_data.pkl"):
        with open(filepath, "rb") as f:
            self.test_data = pickle.load(f)
        return self

    def _extract_sequences_from_folder(self, ordner_pfad):
        """
        Hilfsfunktion: Holt aus allen .pkl Dateien eines Ordners die längsten Trajektorien.
        """
        sequenzen = []
        for dateiname in os.listdir(ordner_pfad):
            if not dateiname.endswith(".pkl"):
                continue

            pfad = os.path.join(ordner_pfad, dateiname)
            try:
                with open(pfad, "rb") as f:
                    recording = pickle.load(f)
            except Exception:
                continue

            if 'preprocessor' not in recording:
                continue

            beste_trajektorie = None
            max_laenge = 0

            # Längste Sequenz finden
            for frame_dict in recording['preprocessor']:
                if frame_dict is None:
                    continue
                daten = frame_dict.get('preprocessor')
                if daten is not None and len(daten) > max_laenge:
                    max_laenge = len(daten)
                    beste_trajektorie = daten

            if beste_trajektorie is not None:
                sequenzen.append(beste_trajektorie)

        return sequenzen

    def fit(self, data_directory, test_size=0.2):
        """
        TODO: Trainiere den Klassifikator

        Ziel:
        -----
        Trainiere ein separates HMM für jede Klasse basierend auf den
        gegebenen Sequenzen.


        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Daten so, dass du pro Klasse alle Sequenzen bekommst
        - Trainiere ein Modell pro Klasse
        - Speichere die trainierten Modelle intern

        .. tip::
           Überlege dir eine sinnvolle Datenstruktur wie:
           ``label -> (Daten, Sequenzlängen)``

        .. note::
           Die konkrete Umsetzung ist offen:
            - Wie genau du Daten aufteilst
            - Wie du dein Modell initialisierst
            - Welche Hyperparameter du verwendest

        .. warning::
           Achte darauf, dass:
            - ``lengths`` zu ``X`` passen
            - Labels korrekt zu Sequenzen zugeordnet sind

        Erweiterung:
        ------------
        - Experimentiere mit verschiedenen Modellgrößen
        - Nutze eine Grid Search zur Optimierung
        - Verwende ein separates Testset zur Evaluation

        Returns
        -------
        self
        """
        print(f"Starte HMM Training. Lade Daten aus: {data_directory}")
        # Klassen aus den Ordnernamen auslesen
        ordner_liste = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
        self.classes_ = sorted(ordner_liste)

        for label in self.classes_:
            ordner_pfad = os.path.join(data_directory, label)
            sequenzen = self._extract_sequences_from_folder(ordner_pfad)

            if len(sequenzen) < 3:
                print(f"  -> Zu wenig Daten für Klasse '{label}'. Überspringe.")
                continue
            random.seed(self.random_state)
            random.shuffle(sequenzen)
            split_idx = int(len(sequenzen) * (1 - test_size))

            train_seqs = sequenzen[:split_idx]
            test_seqs = sequenzen[split_idx:]
            self.test_data[label] = test_seqs

            X_train = np.vstack(train_seqs)
            lengths_train = [len(seq) for seq in train_seqs]

            print(f"  -> Trainiere '{label}': {len(train_seqs)} Train, {len(test_seqs)} Test.")

            # Modell initialisieren und trainieren
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            model.fit(X_train, lengths_train)

            # Trainiertes Modell im Dictionary ablegen
            self.models[label] = model
            print(f"Training abgeschlossen für Klasse {label}!")
        return self

    def decision_function(self,sequences):
        """
        TODO: Berechne Scores für jede Klasse

        Ziel:
        -----
        Berechne für jede Eingabesequenz einen Score pro Klasse
        (z. B. Log-Likelihood unter jedem Modell).

        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Eingabe in einzelne Sequenzen
        - Berechne für jede Sequenz:
            Score unter jedem Klassenmodell
        - Gib eine Struktur zurück wie:
            ``(n_sequences, n_classes)``

        .. tip::
           Die meisten HMM-Implementierungen bieten eine
           ``score``-Funktion für Likelihoods.

        .. note::
           Du entscheidest selbst:
            - Welcher Score verwendet wird
            - Wie du mehrere Sequenzen behandelst

        .. warning::
           Stelle sicher, dass:
            - Die Reihenfolge der Klassen konsistent ist
            - Scores vergleichbar sind

        Returns
        -------
        scores : array-like
            Score pro Sequenz und Klasse
        """
        all_scores = []

        for seq in sequences:
            seq_scores = []

            for label in self.classes_:
                model = self.models.get(label)
                if model is None:
                    seq_scores.append(-float('inf'))
                    continue

                try:
                    # model.score berechnet die Log-Likelihood
                    score = model.score(seq)
                    seq_scores.append(score)
                except Exception:
                    # Falls die Sequenz kaputt oder zu kurz ist
                    seq_scores.append(-float('inf'))

            all_scores.append(seq_scores)

            # Gib ein Array der Form (n_sequences, n_classes) zurück
        return np.array(all_scores)

    def predict(self, sequences):
        """
        TODO: Sage Klassenlabels voraus

        Ziel:
        -----
        Weise jeder Eingabesequenz ein Label zu.

        Anforderungen / Ideen:
        ----------------------
        - Nutze deine ``decision_function``
        - Wähle für jede Sequenz die Klasse mit bestem Score

        .. tip::
           Typischerweise:
           ``argmax über Klassen``

        .. note::
           Achte darauf, dass:
            - Klassenreihenfolge konsistent ist
            - Rückgabewerte klar interpretierbar sind

        Erweiterung:
        ------------
        - Gib zusätzlich Unsicherheiten oder Scores zurück
        - Implementiere Top-k Vorhersagen

        Returns
        -------
        labels : list
            Vorhergesagte Labels
        """
        scores = self.decision_function(sequences)

        predictions = []
        for score_row in scores:

            best_idx = np.argmax(score_row)
            best_label = self.classes_[best_idx]
            predictions.append(best_label)

        return predictions

if __name__ == "__main__":
    daten_pfad = r"C:\Users\Evran\GestureRecognitionMPT\recordings"
    classifier = HMMClassifier(n_components=5)
    classifier.fit(data_directory=daten_pfad)
    classifier.save_model("trained_models/hmm_models.pkl")
    classifier.save_test_data("test_data/test_data.pkl")
