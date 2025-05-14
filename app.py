'''import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('best_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

IMG_SIZE = 128

def draw_fancy_box(img, pt1, pt2, color, label):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw fancy corners
    thickness = 2
    line_length = 30
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)

    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)

    # Add translucent label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x1, y1 - 30), (x1 + text_w + 10, y1), color, -1)
    cv2.putText(img, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while True:
    _, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        pred = model.predict(face)[0][0]

        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        draw_fancy_box(frame, (x, y), (x + w, y + h), color, label)

    cv2.imshow('Real-Time Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('best_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

IMG_SIZE = 128

def draw_stylish_box(frame, pt1, pt2, color, label):
    overlay = frame.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    # Box thickness and corner radius
    thickness = 2
    radius = 20
    line_length = 40

    # Top-left corner
    cv2.line(overlay, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(overlay, (x1, y1), (x1, y1 + line_length), color, thickness)

    # Top-right corner
    cv2.line(overlay, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(overlay, (x2, y1), (x2, y1 + line_length), color, thickness)

    # Bottom-left corner
    cv2.line(overlay, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(overlay, (x1, y2), (x1, y2 - line_length), color, thickness)

    # Bottom-right corner
    cv2.line(overlay, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(overlay, (x2, y2), (x2, y2 - line_length), color, thickness)

    # Add translucent label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_bg = (x1, y1 - 35, x1 + text_w + 20, y1 - 5)
    cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), color, -1)
    cv2.putText(overlay, label, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Blend overlay with frame for transparency effect
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        pred = model.predict(face)[0][0]

        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        draw_stylish_box(frame, (x, y), (x + w, y + h), color, label)

    cv2.imshow('Real-Time Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




