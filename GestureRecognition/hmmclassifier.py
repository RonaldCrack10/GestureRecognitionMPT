import pickle
import numpy as np
from hmmlearn import hmm


class HMMClassifier:
    """
    HMM-basierter Klassifikator für Gestensequenzen.

    Trainiert ein GaussianHMM pro Klasse und klassifiziert
    neue Sequenzen anhand der höchsten Log-Likelihood.

    Parameters
    ----------
    n_components : int
        Anzahl der versteckten Zustände pro HMM.
    n_iter : int
        Maximale Anzahl der EM-Iterationen beim Training.
    random_state : int
        Seed für Reproduzierbarkeit.
    test_size : float
        Anteil der Daten für den Testset (0.0 - 1.0).
    """

    def __init__(self, n_components: int = 5, n_iter: int = 100,
                 random_state: int = 42, test_size: float = 0.2):
        self.n_components  = n_components
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.test_size     = test_size

        self.models_  : dict = {}   # label -> GaussianHMM
        self.classes_ : list = []

   

    def fit(self, X: np.ndarray, lengths: list, labels: list) -> "HMMClassifier":
        """
        Trainiert ein GaussianHMM pro Klasse.

        Parameters
        ----------
        X       : np.ndarray, shape (n_frames_total, n_features)
            Alle Sequenzen zusammenhängend (hmmlearn-Format).
        lengths : list[int]
            Länge jeder einzelnen Sequenz.
        labels  : list[str]
            Label jeder Sequenz — muss dieselbe Länge wie ``lengths`` haben.
        """
        assert len(lengths) == len(labels), "lengths und labels müssen gleich lang sein"

        self.classes_ = sorted(set(labels))

        # Sequenzen nach Label gruppieren
        class_seqs: dict = {c: [] for c in self.classes_}
        idx = 0
        for length, label in zip(lengths, labels):
            class_seqs[label].append(X[idx : idx + length])
            idx += length

        rng = np.random.default_rng(self.random_state)

        for label in self.classes_:
            seqs  = class_seqs[label]
            n     = len(seqs)

            if n < 2:
                print(f"  ✗ '{label}': zu wenig Sequenzen ({n}) — übersprungen")
                continue

            # Train / Test Split
            perm    = rng.permutation(n) # permutation der Indizes für zufällige Aufteilung des Datensatzes
            n_test  = max(1, int(n * self.test_size))
            n_train = n - n_test

            train_seqs = [seqs[i] for i in perm[:n_train]]
            test_seqs  = [seqs[i] for i in perm[n_train:]]

            X_train      = np.concatenate(train_seqs)
            lens_train   = [len(s) for s in train_seqs]

            model = hmm.GaussianHMM(
                n_components    = self.n_components,
                covariance_type = "diag",
                n_iter          = self.n_iter,
                random_state    = self.random_state,

                min_covar=1e-2  # Verhindert zu kleine Varianzen, die zu Singularitäten führen können
            )
            model.fit(X_train, lens_train)
            self.models_[label] = model

            # Evaluation auf Trainings- und Testdaten
            train_ll = model.score(X_train, lens_train) / sum(lens_train)
            X_test   = np.concatenate(test_seqs)
            lens_test = [len(s) for s in test_seqs]
            test_ll  = model.score(X_test, lens_test) / sum(lens_test)

            print(f"  ✓ '{label}':  {n_train} train / {n_test} test  |  "
                  f"train ll: {train_ll:.3f}  test ll: {test_ll:.3f}")

        print(f"\nTraining abgeschlossen — {len(self.models_)} Modelle trainiert.")
        return self


    def decision_function(self, X: np.ndarray, lengths: list) -> np.ndarray: # -> shape (n_sequences, n_classes)
        """
        Berechnet Log-Likelihood jeder Sequenz unter jedem Klassenmodell.

        Parameters
        ----------
        X       : np.ndarray  - Sequenzen im hmmlearn-Format
        lengths : list[int]   - Länge jeder Sequenz

        Returns
        -------
        scores : np.ndarray, shape (n_sequences, n_classes)
        """
        n_seq  = len(lengths)
        n_cls  = len(self.classes_)
        scores = np.full((n_seq, n_cls), -np.inf) # till infinity because log-likelihoods can be very negative and we want to avoid numerical issues

        for j, label in enumerate(self.classes_):
            model = self.models_.get(label)
            if model is None:
                continue

            idx = 0
            for i, length in enumerate(lengths):
                seq = X[idx : idx + length]
                try:
                    scores[i, j] = model.score(seq) / length  # Normalize by sequence length
                except Exception:
                    scores[i, j] = -np.inf
                idx += length

        return scores
    '''
    decision_function funktioniert so dass sie für jede Sequenz (Zeile) 
    die Log-Likelihood unter jedem Klassenmodell (Spalte) berechnet. 
    Das Ergebnis ist eine 2D-Array, in der der höchste Wert in jeder Zeile angibt, welches Modell 
    die Sequenz am besten erklärt.

    log-likelihoods wird angewendet, weil die Wahrscheinlichkeit von Sequenzen unter HMMs oft sehr 
    klein ist, und die Logarithmierung hilft, mit diesen kleinen Zahlen umzugehen und 
    numerische Stabilität zu gewährleisten.
    '''
    def predict(self, X: np.ndarray, lengths: list) -> list:
        """
        Gibt für jede Sequenz das wahrscheinlichste Label zurück.

        Parameters
        ----------
        X       : np.ndarray
        lengths : list[int]

        Returns
        -------
        labels : list[str]
        """
        scores  = self.decision_function(X, lengths)
        indices = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in indices]

    def predict_single(self, seq: np.ndarray) -> str:
        """
        Klassifiziert eine einzelne Sequenz.

        Parameters
        ----------
        seq : np.ndarray, shape (n_frames, n_features)

        Returns
        -------
        label : str
        """
        return self.predict(seq, [len(seq)])[0]

    

    def evaluate(self, X: np.ndarray, lengths: list, labels: list) -> float:
        """
        Berechnet die Accuracy auf einem Datensatz.

        Parameters
        ----------
        X, lengths, labels : wie bei fit()

        Returns
        -------
        accuracy : float
        """
        preds = self.predict(X, lengths)
        correct = sum(p == t for p, t in zip(preds, labels))
        accuracy = correct / len(labels)

        print(f"\nAccuracy: {correct}/{len(labels)} = {accuracy:.1%}")

        # Konfusionsmatrix ausgeben
        print("\nKonfusionsmatrix:")
        print(f"{'':>6}", end="")
        for c in self.classes_:
            print(f"{c:>6}", end="")
        print()
        for true_c in self.classes_:
            print(f"{true_c:>6}", end="")
            for pred_c in self.classes_:
                count = sum(p == pred_c and t == true_c
                            for p, t in zip(preds, labels))
                print(f"{count:>6}", end="")
            print()

        return accuracy


    def save(self, path: str) -> None:
        """Speichert das trainierte Modell als pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Modell gespeichert → {path}")

    @staticmethod
    def load(path: str) -> "HMMClassifier":
        """Lädt ein gespeichertes Modell."""
        with open(path, "rb") as f:
            clf = pickle.load(f)
        print(f"Modell geladen ← {path}")
        return clf




if __name__ == "__main__":
    import pickle as pkl
    from pathlib import Path

    dataset_path = Path("data/dataset.pickle")
    if not dataset_path.exists():
        print("Kein Dataset gefunden. Erst labeling.py ausführen.")
    else:
        with open(dataset_path, "rb") as f:
            ds = pkl.load(f)

        clf = HMMClassifier(n_components=4, n_iter=100, test_size=0.2)
        clf.fit(ds["X"], ds["lengths"], ds["labels"])
        clf.evaluate(ds["X"], ds["lengths"], ds["labels"])
        clf.save("data/hmm_model.pickle")