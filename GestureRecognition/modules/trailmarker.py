from SignalHub import Module, get_nested_key
from collections import deque

class TrailMarker(Module):
    """
    #test
    Modul zum Zeichnen einer Spur anhand der Bewegung eines Fingers.

    Die Position eines bestimmten Finger-Landmarks wird über mehrere Frames
    hinweg gespeichert. Aus diesen Punkten kann anschließend eine Linie
    erzeugt werden, die den Bewegungsverlauf des Fingers visualisiert.

    Ziel ist es, die Verarbeitung der Landmark-Daten sowie die Verwaltung
    eines Zustands über mehrere Frames hinweg selbst zu implementieren.
    """

    def __init__(self, outputSignal="trailmarker"):
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
        - ``detector`` : Ergebnisse der Handdetektion

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Da dieses Modul keine eigenen Daten erzeugt, reicht beispielsweise:

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
        """
        self.outputSignal = outputSignal

        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle Variablen vorzubereiten, die während der
        Laufzeit des Moduls benötigt werden.

        Hinweise
        --------
        - Lese benötigte Parameter aus der Konfiguration.
        - Bestimme beispielsweise, welcher Finger verfolgt werden soll.
        - Lege eine Datenstruktur an, in der mehrere vergangene
          Fingerpositionen gespeichert werden können,
          z.B. :class:`collections.deque` mit einer maximalen Größe.
        - Diese Historie wird später verwendet, um eine Spur zu zeichnen.
        - Speichere aus der Konfiguration weitere benötigte Parameter,
          z.B. Finger-Index, maximale Anzahl verlorener Frames oder
          Webcam-Parameter.
        - Für den Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
           Eine ``deque`` ist ideal für Trajektorien,
           da sie effizient alte Punkte entfernt.

        .. note::
           Initialisiere hier nur Zustände und Parameter,
           keine eigentliche Verarbeitung.

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks. Enthält unter anderem das
            Signal ``config``.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """

        # lese die config datei ein
        config = data.get("config", {})

        # bestimme den zu verfolgenen finger index
        self.finger_index = get_nested_key(
            config, "trailmarker.fingerIndex", 8
        )

        #bestimme wie lange der FInger verfolgt werden soll
        self.max_points = get_nested_key(
            config, "trailmarker.maxPoints", 50
        )

        #bestimme wie viele frames ohne erkannten Finger erlaubt sind, bevor die Spur gelöscht wird
        self.max_lost_frames = get_nested_key(
            config, "trailmarker.maxLostFrames", 10
        )

        # erstelle die trajektorie
        self.trail = deque(maxlen=self.max_points)
        self.lost_frames = 0

        return {}


    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, die aktuelle Position eines Fingers zu bestimmen,
        diese Position in einer Trajektorie zu speichern und daraus
        eine visuelle Spur zu erzeugen.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte Hände und
          deren Landmarken zu erhalten.
        - Falls keine Hand erkannt wurde, kann beispielsweise ein Zähler
          für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position kann zur bestehenden Trajektorie hinzugefügt werden.
        - Zwischen aufeinanderfolgenden Punkten können Linien gezeichnet
          werden, um eine Spur darzustellen.
        - Für die Visualisierung kann :meth:`line` der :class:`GALY`
          verwendet werden.

        .. tip::
          Typischer Ablauf:
           1. Landmark extrahieren
           2. Punkt speichern
           3. Trajektorie aktualisieren
           4. Linien zwischen Punkten zeichnen

        .. warning::
            Achte darauf, dass:
              - keine leeren Landmark-Daten verarbeitet werden
              - die Trajektorie nicht unendlich wächst
              - verlorene Frames sinnvoll behandelt werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Um die Zeichenoperationen auszuführen, sollte ein
            :class:`GALY` Objekt zurückgegeben werden.

            Beispiel:

            ``return { ..., "galy": galy}``
        """
        detector = data.get("detector", {})
        galy = data.get("galy", None)

        hands = detector.get("hands", [])

        # Nur True wenn keine Hand erkannt wurde, sonst False
        if len(hands) == 0:
            self.lost_frames += 1

            #Nur True wenn über 10 Frames keine Hand erkannt wurde, sonst False
            if self.lost_frames > self.max_lost_frames:
                self.trail.clear()

            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        #----------------------------------------------------------------------
        # Code ist nur hier wenn eine Hand erkannt wurde
        #----------------------------------------------------------------------

        # setze lost_frames zurück, da eine Hand erkannt wurde
        self.lost_frames = 0

        # Nimm die erste erkannte Hand (falls mehrere erkannt wurden)
        hand = hands[0]

        # Speichere die Landmarks der Hand
        landmarks = hand.get("landmarks", [])

        # Breche ab, wenn die Anzahl der Landmarks nicht ausreicht
        if len(landmarks) <= self.finger_index:
            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        # Wähle welcher Finger verfolgt werden soll
        landmark = landmarks[self.finger_index]

        #zwischenspeichere die x und y Koordinaten des Fingers
        x = landmark.get("x")
        y = landmark.get("y")

        # Breche ab falls nicht beide Koordinaten vorhanden sind
        if x is None or y is None:
            return {
                self.outputSignal: {
                    "trail": list(self.trail)
                },
                "galy": galy
            }

        # erstelle ein Tupel mit den x,y Koordinaten
        point = (x, y)

        # Füge das Tupel der trajektorie hinzu
        self.trail.append(point)

        if galy is not None:
            points = list(self.trail)

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                galy.line(x1, y1, x2, y2)

        return {
            self.outputSignal: {
                "trail": list(self.trail)
            },
            "galy": galy
        }


    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber sinnvoll sein,
           wenn Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        self.trail.clear()