import numpy as np
from SignalHub import GALY, bgr, get_nested_key, Module
from GestureRecognition.hmmclassifier import HMMClassifier


class HMMModule(Module):
    """
    Modul zur Klassifikation von Gesten mittels Hidden Markov Models.

    Empfängt eine vorverarbeitete Fingertrajektorie vom Preprocessor,
    klassifiziert sie mit einem trainierten HMMClassifier und
    visualisiert das Ergebnis.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm_model.pickle", **kwargs):
        self.outputSignal = outputSignal
        self.model_path   = model_path

        super().__init__(
            name         = "hiddenmarkov",
            inputSignals = ["config", "preprocessor"],
            outputSchema = {"type": "object", "properties": {outputSignal: {}}},
        )

    def start(self, data: dict) -> dict:
        model_path  = get_nested_key('config.hmm_model_path', data, default=self.model_path)
        self.model  = HMMClassifier.load(model_path)
        self.last_result = None 
        return {}

    def step(self, data: dict) -> dict:
        trajectory = get_nested_key('preprocessor', data)

        if trajectory is not None and len(trajectory) > 0:
            # Preprocessor already normalizes, use data as-is
            seq = np.array(trajectory, dtype=np.float32)

            scores_arr = self.model.decision_function(seq, [len(seq)])[0]
            best_label = self.model.predict_single(seq)
            best_score = float(np.max(scores_arr))

            print(f"seq shape:  {seq.shape}")
            print(f"seq min:    {seq.min():.4f}")
            print(f"seq max:    {seq.max():.4f}")
            print(f"seq mean:   {seq.mean():.4f}")
            print(f"scores:     {dict(zip(self.model.classes_, scores_arr))}")

            self.last_result = {
                "label":  best_label,
                "score":  best_score,
                "scores": dict(zip(self.model.classes_, scores_arr)),
            }
        if self.last_result is None:
            return {}

        width  = get_nested_key('config.width',  data, default=1280)
        height = get_nested_key('config.height', data, default=720)

        galy = GALY()
        galy.layer("hmm")

        galy.putText(
            f"{self.last_result['label']}  {self.last_result['score']:.2f}",
            (int(width * 0.05), int(height * 0.1)),
            fontScale = 1.5,
            color     = bgr("#FC0000"),
        )
        
        return {self.outputSignal: self.last_result, "galy": galy}

    
    def stop(self, data: dict) -> None:
        pass