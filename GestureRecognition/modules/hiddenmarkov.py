from SignalHub import GALY, bgr, get_nested_key, Module, HMM


class HMMModule(Module):
    """
    Modul zur Klassifikation von Gesten mittels Hidden Markov Models.

    Dieses Modul erhält eine vorverarbeitete Fingertrajektorie vom
    :class:`Preprocessor` Modul und verwendet ein trainiertes
    Hidden-Markov-Modell, um eine Geste zu klassifizieren.

    Ziel ist es, eine geladene Modellstruktur zu verwenden, um
    eine Entscheidung über die aktuell ausgeführte Bewegung zu treffen
    und das Ergebnis an das Framework zurückzugeben.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``preprocessor`` : normalisierte Trajektorien

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``markov``.

        Dieses Signal enthält Informationen über die erkannte Geste
        sowie deren Klassifikationsscore.

        Beispiel:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.

        model_path : str, optional
            Pfad zu einem gespeicherten HMM-Modell.

        **kwargs
            Weitere Parameter, die an :class:`Module` weitergegeben werden.
        """
        self.OutputSignal = outputSignal
        self.model_path = model_path
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )

    def start(self, data):
        """
        Initialisierung des Moduls.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, ein zuvor trainiertes Hidden-Markov-Modell zu laden,
        das später zur Klassifikation verwendet wird.

        Hinweise
        --------
        - Das Modell kann aus einer Datei geladen werden.
        - Typischerweise wird dafür eine Klassenmethode verwendet,
          die ein gespeichertes Modell rekonstruiert.
        - Das geladene Modell sollte als Attribut des Moduls gespeichert
          werden, damit es in :meth:`step` verwendet werden kann.

        .. tip::
           Trenne klar zwischen:
            - Modell laden (``start``)
            - Modell anwenden (``step``)

        .. warning::
           Stelle sicher, dass:
            - der Pfad korrekt ist
            - das Modell zum erwarteten Datenformat passt

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        model_path = get_nested_key(data, 'config.hmm_model_path', default= self.model_path) 
        self.model = HMM.load(model_path)
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, eine vorverarbeitete Trajektorie zu klassifizieren
        und die wahrscheinlichste Geste zu bestimmen.

        Hinweise
        --------
        - Greife auf das ``preprocessor`` Signal zu.
        - Falls keine Trajektorie vorhanden ist, kann die Verarbeitung
          übersprungen werden.
        - Das geladene HMM-Modell kann anschließend verwendet werden,
          um eine Entscheidung für die aktuelle Bewegung zu berechnen.
        - Das Ergebnis enthält typischerweise Scores für mehrere Klassen.
        - Die Klasse mit dem höchsten Score kann als Ergebnis gewählt werden.

        Zusätzlich kann eine Visualisierung erzeugt werden:

        - Erzeuge ein :class:`GALY` Objekt.
        - Lege eine neue Zeichenebene an.
        - Verwende :meth:`putText`, um Score und Label darzustellen.
        - Für die Skalierung der Zeichenebene können Parameter aus der
          Konfiguration über :meth:`get_nested_key` gelesen werden.

        .. tip::
           Typischer Ablauf:
            1. Daten prüfen (existiert eine Sequenz?)
            2. Modell anwenden
            3. Scores interpretieren
            4. Ergebnis visualisieren

        .. note::
           Du entscheidest selbst:
            - wie du Scores darstellst
            - ob du nur das beste Label oder mehrere Kandidaten zeigst

        .. warning::
           Achte darauf, dass:
            - das Eingabeformat exakt zum Trainingsformat passt
            - keine leeren oder fehlerhaften Sequenzen verarbeitet werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``preprocessor`` : normalisierte Trajektorie
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Soll die erkannte Geste sowie optional Visualisierungsdaten
            enthalten.

            Beispiel:

            ``return {outputSignal: result, "galy": galy}``
        """
        trajectory = get_nested_key(data, 'preprocessor')
        if trajectory is None:
            return {}
        
        # score für jede klasse berechnen
        score = self.model.predict(trajectory)

        # beste klasse bestimmen
        best_label = max(score, key = score.get)
        best_score = score[best_label]
        results = {"label": best_label, "score": best_score, "scores": score}

        galy = GALY()
        layer = galy.new_layer()

        width  = get_nested_key(data, 'config.width',  default=640)
        height = get_nested_key(data, 'config.height', default=480)

        galy.putText(
            layer,
            f"{best_label}: {best_score:.2f}",
            (int(width * 0.05), int(height * 0.1)),
            scale=1.5,
            color=bgr("#00FF00")
        )

        return {self.OutputSignal: results, 'galy': galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf interne Zustände zurückzusetzen
        oder Ressourcen freizugeben.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber relevant werden,
           wenn Modelle oder externe Ressourcen verwaltet werden.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass