---
layout: default
title: Auto-Director
parent: Anwendungsfälle
nav_order: 2
---



# Auto-Director

Es gibt verschiedene Möglichkeiten, wie YOLO implementiert werden kann, um verschiedene Objekte auf Standbildern, Videos oder Echtzeit-Kameramaterial zu erkennen. Der schnellste Weg für die Erkennung, leistungstechnisch, besteht darin, YOLO mit darknet unter Verwendung von CUDA zu implementieren. Dies ist auch der komplexeste Weg, YOLO auf Webcam-Material anzuwenden. Im Rahmen des Workshops werden wir jedoch eine etwas langsamere Implementierung verwenden und YOLO mit OpenCV in Python ausführen.

Eine funktionierende Installation von Python wird vorausgesetzt.

## 0. Installation zusätzlicher Pakete mit PIP

Python hat nicht standardmäßig alle Pakete mitgeliefert, die für eine Objekterkennung mit YOLO notwendig sind.

Fehlende Pakete müssen mit PIP, dem Paketinstallations- und -managementprogramm von Python vorher über die Commandline bzw. das Terminal installiert werden.

Die für die Echtzeit-Objekterkennung in Python benötigten Pakete sind:

- OpenCV: `pip install opencv-python`
- Ultralytics Implementierung von YOLO: `pip install ultralytics`

## 1. Abrufen eines Bildstroms von einer Webcam und Anzeigen

Öffnen Sie die Python-Datei `autodirector_tutorial.py` in Ihrer IDE und fügen Sie die nachfolgenden Code-Schnipsel an den entsprechenden Stellen ein.

Um mithilfe von OpenCV 2 die Bilder der Kamera darzustellen und im späteren Verlauf zu verändern, müssen Sie zwei Pakete und ihre Funktionen zuerst importieren:

```python
# 1.0
import cv2
import math 
```

Die Webcam wird mit OpenCV wie folgt gestartet.

Dieser Code erstellt ein `VideoCapture`-Objekt für die Erfassung von Frames von der Standardkamera (0). Die Auflösung der Webcam wird auf 640x480 festgelegt. 

Beachten Sie, dass die `0` als Parameter in der Funktion `cv2.VideoCapture(0)` auf die ERSTE gefundene Kamera zu greift. Bei mehreren Kameras müssen Sie hier entsprechend die gewünschte Kamera spezifizieren.

Genauso wie die Breite und Höhe der Webcam je nach Auflösung angepasst werden muss.

```python
# 1.1
# Webcam initialisieren
cap = cv2.VideoCapture(0)
# Breite und Höhe der Webcam spezifizieren
cap.set(3, 640)
cap.set(4, 480)
```

Anschließend setzen wir das Programm in eine Endlosschleife und zeigen fortwährend die Frames der Kamera in einem Fenster an.

Wenn der Benutzer `q` drückt, wird die Ausührung beendet.

```python
# 1.2
# Diese Schleife läuft für immer (bis wir explizit sagen es gibt einen "break")
# Dadurch wird das Kamerabild durchgehend geladen, verarbeite und angezeigt
while True:
    # Hier greifen wir mit OpenCV auf das Kamerabild zu und speichern es in der Variable "img"
    success, img = cap.read()
    # 2.3
    
    # Hier zeigen wir das Webcam-Bild ("img") in einem Fenster
    cv2.imshow('Webcam', img)
    # WENN wir die Taste "q" drücken, beenden wir die Schleife
    # So lässt sich das Programm beenden
    if cv2.waitKey(1) == ord('q'):
        break

# Wenn wir das Programm beenden, löschen wir den Zugriff auf die Webcam und schließen das Fenster
cap.release()
cv2.destroyAllWindows()
```

## 2. Objekterkennung mit YOLO

Um mit der Objekterkennung zu beginnen, importieren Sie die Implementierung von Ultralytics YOLO:

```python
# 2.0
from ultralytics import YOLO
```

Nun laden wir ein spezifischen YOLO-Model und speichern es in der Variable `model`

```python
# 2.1
# YOLO Model laden
model = YOLO("yolo-Weights/yolov8n.pt")
```

Erstellen Sie dann eine Variable `classNames`, die eine Liste mit allen möglichen Klassennamen von Objekten enthält. Dies erleichtert später das Referenzieren bestimmter Klassen:

```python
# 2.2
# Namen der klassifizierbaren Objekte speichern
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

```

Nachdem das Webcam-Bild mit `success, img = cap.read()` gelesen wurde, verwenden wir YOLO, um Objekte zu erkennen und sie in unserer Ergebnisliste `results` zu speichern:

```python
    # 2.3
    # An dieser Stelle verwenden wir YOLO um das Kamerabild ("img") zu analysieren
    # Das Resultat wird in der Variable "results" gespeichert
    results = model(img, stream=True)
    # 3.0
```

## 3. Auswertung der Ergebnisse

Um die Ergebnisse (erkannten Objekte) auszuwerten, durchlaufen Sie die Ergebnisse mit zwei Schleifen:

```python
    # 3.0
    # Jetzt durchlaufen wir die Resultate und schauen uns jedes erkannte Objekt an
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 3.1

            # 3.2
            
            # 3.3
```

In diesem Beispiel möchten wir Personen und Handys erkennen.

Im Anschluss nutzen wir diese erkannten Objekte um das Webcam-Bild abhängig von ihren Positionen zu verändern.

Dafür ist wichtig herauszufinden, wie sicher sich YOLO ist ein bestimmtes Objekt zu erkennen. Auf diesen Wert greifen Sie zu und speichern ihn in der Variable `confidence`

Desweiteren greifen Sie auf die Klasse des erkannten Objekts zu um nachher den Bildausschnitt verändern zu könnnen, je nachdem ob ein Mensch und/oder ein Handy im Kamerabild erkannt wurde.

```python
            # 3.1
            # "confidence" beschreibt wie sehr YOLO sich sicher ist, dass das Objekt zu einer bestimmten Klasse gehört
            # Die "confidence" liegt zwischen 0 (0%) und 1 (100%)
            confidence = math.ceil((box.conf[0]*100))/100

            # Hier speichern wir den Index der Klassifizierung zwischen um ihn später auswerten zu können
            cls = int(box.cls[0])
```

Dies ist die Logik für Erkennung einer Person:

Sollte YOLO sich zu 50% sicher sein, dass sich in einem Bildausschnitt eine Person befindet, speichern Sie den Begrenzungsrahmen dieser.

```python
            # 3.2
            # Sollte YOLO sich zu mehr als 50% sicher sein, dass ein Objekt der Klasse "Person" gefunden wurde, machen wir etwas
            if confidence >= 0.5 and classNames[cls] == "person":
                # Hier finden wir die "bounding box" der Person, den Rahmen des Bildausschnitts indem sie sich befindet
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Konvertieren in Integer (Pixel können keine Nachkommastellen haben!)

                # 4.2
                # 4.1
```

Dies ist die Logik für Erkennung eines Handys:

Sollte YOLO sich zu 50% sicher sein, dass sich in einem Bildausschnitt ein Handy befindet, speichern Sie den Begrenzungsrahmen dieser.

```python
            # 3.3
            # Sollte YOLO sich zu mehr als 50% sicher sein, dass ein Objekt der Klasse "cell phone" gefunden wurde, machen wir etwas
            if confidence >= 0.5 and classNames[cls] == "cell phone":
                # Hier finden wir die "bounding box" des Handys, den Rahmen des Bildausschnitts indem sie sich befindet
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Konvertieren in Integer (Pixel können keine Nachkommastellen haben!)
                # 4.3
```

## 4. Verändern des Bildes in Echtzeit

Jetzt besitzen Sie alle notwendigen Informationen um die Frames der Webcam je nach den ausgewerteten Ergebnissen zu verändern.

Wir werden drei Szenarien implementieren, die Sie an- und ausschalten können:

1. Ein Namensschild über der Person im Bildausschnitt
2. Das automatische Pixeln von Personen im Bildausschnitt
3. Ein Logo einfügen anstelle eines Handys im Bildausschnitt

Der Einffachheit wegen, erstellen wir drei Variablen zu Beginn als "Einstellungen" um jede Funktionen an- und ausschalten zu können.

```python
# 4.0
# Diese Variablen sind unsere Einstellungen, sie aktivieren verschiedene Funktionen nach der Bildanalyse mit YOLO
blur = False
name = True
phoneLogo = True
```

Um einen Namen überhalb einer Person einblenden zu können, brauchen Sie die Position der Person im Bildausschnit.

Die "Bounding Box" der erkannten Person haben Sie bereits gespeichert. Anhand dieser können Sie die horizontale Mitte einer Person berechnen.

Im Anschluss fügen wir auf diese Position im Bild den Namen ein.

```python
                # 4.1
                # Wenn wir die Einstellung "name" aktiviert haben, setzen wir ein Namensschild auf den Kopf der Person
                if name:
                    # Die Position der Schrift (in der horizontalen Mitte und an der obersten Kante des Ausschnitts der Person)
                    location = [int(x1 + (x2 - x1)/2.0), y1]
                    # Die Schriftart
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Die Größe der Schriftart
                    fontScale = 1
                    # Die Farbe der Schrift
                    color = (255, 0, 0)
                    # Die Breite der Schrift
                    thickness = 2
                    # Und zuletzt setzen wir das Wort ("EIN NAME") auf das Bild unter Berücksichtigung der Schrift-Paramater
                    cv2.putText(img, "EIN NAME", location, font, fontScale, color, thickness)
```

Wenn YOLO Personen erkennt, sollen diese automatisch im Bild verpixelt werden.

Hierfür greifen Sie auch wieder auf die gespeicherte Bounding Box zu, schneiden diesen Ausschnitt aus, zeichnen ihn weich und fügen ihn anschließend wieder zurück in das Webcam-Bild.

```python
                # 4.2
                # Wenn wir die Einstellung "blur" aktiviert haben, verpixeln wir nun die Person
                if blur:
                    # Zuerst spezifizieren wir noch einmal mehr den Bildausschnitt in den sich die Person befindet
                    topLeft = (x1, y1)
                    bottomRight = (x2, y2)
                    x, y = topLeft[0], topLeft[1]
                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

                    # Nun schneiden wir diesen Ausschnitt aus dem Webcam-Bild aus
                    ROI = img[y:y+h, x:x+w]
                    # Und zeichnen ihn mit einem Gaussian Blur weich
                    blur = cv2.GaussianBlur(ROI, (51,51), 0) 

                    # Zuletzt fügen wir ihn wieder in das Webcam-Bild ein
                    img[y:y+h, x:x+w] = blur
```

Zuletzt sollen erkannte Handys durch ein Logo ersetzt werden.

Dafür müssen Sie zuerst das gewünscht Logo mit `cv2.imread` laden und zwischenspeichern.

Im Anschluss greifen Sie auch wieder auf die gespeicherte Bounding Box zu, skalieren das Logo auf die Größe dieser, ersetzen diesen Ausschnitt durch das Logo und fügen ihn anschließend wieder zurück in das Webcam-Bild.

```python
                # 4.3
                # Wenn wir die Einstellung "phoneLogo" aktiviert haben, ersetzen wir ein Smartphone durch ein Logo
                if phoneLogo:
                    # Wir lesen das Logo ein und speichern es unter der Variable "logo_color"
                    logo_color = cv2.imread('hmc-logo-quer-schwarz.jpg')
                    # Dann spezifizieren wir noch einmal mehr den Bildausschnitt in den sich das Handy befindet
                    topLeft = (x1, y1)
                    bottomRight = (x2, y2)
                    x, y = topLeft[0], topLeft[1]
                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                    # Nun passen wir die Logo-Größe auf die Handygröße an
                    imageSize = (w, h)
                    logo_color = cv2.resize(logo_color, imageSize, interpolation= cv2.INTER_LINEAR)
                    # Und fügen das Logo in den Bildausschnitt ein, in dem sich das Handy befindet
                    img[y:y+h, x:x+w] = logo_color
```

## Der vollständige Code
```python
from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    blur = False
    name = True
    phoneLogo = True

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", classNames[cls])
            if confidence >= 0.5 and classNames[cls] == "person":
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # blur box on screen
                if blur:
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    # Create ROI coordinates
                    topLeft = (x1, y1)
                    bottomRight = (x2, y2)
                    x, y = topLeft[0], topLeft[1]
                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

                    # Grab ROI with Numpy slicing and blur
                    ROI = img[y:y+h, x:x+w]
                    blur = cv2.GaussianBlur(ROI, (51,51), 0) 

                    # Insert ROI back into image
                    img[y:y+h, x:x+w] = blur

                if name:
                    # object details
                    location = [int(x1 + (x2 - x1)/2.0), y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, "Malte", location, font, fontScale, color, thickness)

            if confidence >= 0.5 and classNames[cls] == "cell phone":
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                if phoneLogo:
                    # Logo to display
                    logo_color = cv2.imread('hmc-logo-quer-schwarz.jpg')
                    # Create ROI coordinates
                    topLeft = (x1, y1)
                    bottomRight = (x2, y2)
                    x, y = topLeft[0], topLeft[1]
                    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                    # object details
                    imageSize = (w, h)
                    logo_color = cv2.resize(logo_color, imageSize, interpolation= cv2.INTER_LINEAR)
                    # Insert Logo into image
                    img[y:y+h, x:x+w] = logo_color
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```