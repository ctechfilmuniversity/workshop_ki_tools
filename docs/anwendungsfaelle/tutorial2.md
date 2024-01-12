---
layout: default
title: Anwendungsfall 2 - Auto-Director
parent: Anwendungsfaelle
nav_order: 2
---



# Anwendungsfall 2: Auto-Director

There are several ways YOLO can be implemented to detect different objects of interest on still images, videos or real time camera footage.

The fastest way for detection, performance-wise,  is building YOLO with darknet using CUDA.

It is also the most complex way to run YOLO on webcam footage.

For ease of use within the scope of the workshop, we will use a slightly slower implementation and run YOLO with OpenCV in Python.

A working installation of Python will be assumed.

### 1. Installing additional packages with PIP
Packages needed for real-time object detection using Python are:
OpenCV: 
```python
pip install opencv-python

```
Ultralytic’s implementation of YOLO: 

```python
pip install ultralytics

```
### 2. Getting a feed of images from a webcam and displaying it
1. Create a new Python file in your IDE and insert the following code:

```python
import cv2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)
       
    if cv2.waitKey(1) == ord('q'):
        break
                   
cap.release()
cv2.destroyAllWindows()
```

This will create a VideoCapture object and set it to capture frames from the default camera (0). It sets the resolution of the web cam to 640x480. 

We then loop through the frames and display them in a window until the user presses ‘q’ to exit.

### 3. Detecting objects with YOLO
Import Ultralytic’s implementation of YOLO:

```python
from ultralytics import YOLO
model = YOLO("yolo-Weights/yolov8n.pt")

```

Then create a variable holding a list with all the prossible objects’ classnames.This makes it easier for us to reference specific classes later:

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
Now, after reading the webcam image with “ret, img= cap.read()”, we will use YOLO to detect objects and save them in our list of results:
```python
 results = model(img, stream=True)

```
### 4. Evaluating results and establishing a threshold
To evaluate the results (detected objects), we will loop over the results with:

```python
for r in results:
    boxes = r.boxes
            
for box in boxes:

```
In this example, we want to only detect people and also do something when there are more than two people detected in the frame.
So first, we need a need a variable that acts as a counter for the amount of people in frame, initialized before the for loop:
 
```python
nPersons = 0

```
Now, if YOLO detects an object with the class name “person” and is more than 50% sure that it is right, we will want to draw a box around the object on frame.
This code goes into the for box loop:

```python
# confidence
confidence = math.ceil((box.conf[0]*100))/100

# class name
cls = int(box.cls[0])

if confidence >= 0.5 and classNames[cls] == "person":
    nPersons += 1
    # bounding box
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    # put box in cam
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

```
Lastly, if there are at least two people, we will print a special message to the console.
This could also trigger a scene change in OBS or something similar.

```python
if nPersons >= 2:
        print(nPersons, " persons detected.")

```
The full code
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

    nPersons = 0

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            if confidence >= 0.5 and classNames[cls] == "person":
                nPersons += 1
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    if nPersons >= 2:
        print(nPersons, " persons detected.")
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
