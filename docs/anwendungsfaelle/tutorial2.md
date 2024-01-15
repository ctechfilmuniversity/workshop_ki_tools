---
layout: default
title: Auto-Director
parent: Anwendungsfälle
nav_order: 2
---



# Auto-Director

Es gibt verschiedene Möglichkeiten, wie YOLO implementiert werden kann, um verschiedene Objekte auf Standbildern, Videos oder Echtzeit-Kameramaterial zu erkennen. Der schnellste Weg für die Erkennung, leistungstechnisch, besteht darin, YOLO mit darknet unter Verwendung von CUDA zu implementieren. Dies ist auch der komplexeste Weg, YOLO auf Webcam-Material anzuwenden. Im Rahmen des Workshops werden wir jedoch eine etwas langsamere Implementierung verwenden und YOLO mit OpenCV in Python ausführen.

Eine funktionierende Installation von Python wird vorausgesetzt.

## 1. Installation zusätzlicher Pakete mit PIP

Die für die Echtzeit-Objekterkennung in Python benötigten Pakete sind:

- OpenCV: `pip install opencv-python`
- Ultralytics Implementierung von YOLO: `pip install ultralytics`

## 2. Abrufen eines Bildstroms von einer Webcam und Anzeigen

Erstellen Sie eine neue Python-Datei in Ihrer IDE und fügen Sie den folgenden Code ein:

```python
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
Dieser Code erstellt ein `VideoCapture`-Objekt für die Erfassung von Frames von der Standardkamera (0). Die Auflösung der Webcam wird auf 640x480 festgelegt. Anschließend durchlaufen wir die Frames und zeigen sie in einem Fenster an, bis der Benutzer 'q' drückt, um die Ausührung zu beenden.

## 3. Objekterkennung mit YOLO

Importieren Sie die Implementierung von Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("yolo-Weights/yolov8n.pt")
```
Erstellen Sie dann eine Variable, die eine Liste mit allen möglichen Klassennamen von Objekten enthält. Dies erleichtert später das Referenzieren bestimmter Klassen:

```python
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
Nachdem das Webcam-Bild mit `success, img = cap.read()` gelesen wurde, verwenden wir YOLO, um Objekte zu erkennen und sie in unserer Ergebnisliste zu speichern:

```python
while True:
    success, img = cap.read()
    results = model(img, stream=True)
```

## 4. Auswertung der Ergebnisse und automatisches Pixeln

Um die Ergebnisse (erkannten Objekte) auszuwerten, durchlaufen wir die Ergebnisse mit:
```python
    for r in results:
        boxes = r.boxes
        for box in boxes:
```

In diesem Beispiel möchten wir nur Personen erkennen und diese automatisch im Bild pixeln.

Wenn YOLO ein Objekt mit dem Klassennamen "person" erkennt und zu mehr als 50% sicher ist, dass es richtig ist, möchten wir diesen Ausschnitt aus dem Bild weichzeichnen. 

Dieser Code geht in die Schleife `for box`:

```python
            # Vertrauen
            confidence = math.ceil((box.conf[0]*100))/100

            # Klassenname
            cls = int(box.cls[0])
            if confidence >= 0.5 and classNames[cls] == "person":
                nPersons += 1
                # Begrenzungsrahmen
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # in Integer-Werte umwandeln
                topLeft = (x1, y1)
                bottomRight = (x2, y2)
                x, y = topLeft[0], topLeft[1]
                w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                # Nimm die ROI mit Numpy slicing und zeichne sie weich
                ROI = img[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(ROI, (51,51), 0) 

                # FÜge die weichgezeichnete ROI wieder zurück in das Bild
                img[y:y+h, x:x+w] = blur
```

Zuletzt zeigen wir das Bild mit dem weichgezeichneten Ausschnitt und ermöglichen die Ausführung mit der Taste 'q' zu stoppen:

```python
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Der vollständige Code
```python
from ultralytics import YOLO
import cv2
import math 

# Webcam starten
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Modell
model = YOLO("yolo-Weights/yolov8n.pt")

# Klassen von Objekten
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

    # Koordinaten
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Vertrauen
            confidence = math.ceil((box.conf[0]*100))/100

            # Klassenname
            cls = int(box.cls[0])
            if confidence >= 0.5 and classNames[cls] == "person":
                nPersons += 1
                # Begrenzungsrahmen
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # in Integer-Werte umwandeln
                topLeft = (x1, y1)
                bottomRight = (x2, y2)
                x, y = topLeft[0], topLeft[1]
                w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
                # Nimm die ROI mit Numpy slicing und zeichne sie weich
                ROI = img[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(ROI, (51,51), 0) 

                # FÜge die weichgezeichnete ROI wieder zurück in das Bild
                img[y:y+h, x:x+w] = blur
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```