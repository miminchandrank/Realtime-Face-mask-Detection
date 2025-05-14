# 😷 Real-Time Face Mask Detection Using CNN

An AI-powered real-time face mask detection system built with **Convolutional Neural Networks (CNNs)** and **OpenCV**, designed to enhance **public safety** during respiratory disease outbreaks such as COVID-19.

🔗 **Live Demo Video**: [Watch on LinkedIn](https://www.linkedin.com/posts/miminchandrank_deeplearning-computervision-ai-activity-7319195495975829504-Ut_o?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFD4aN8BBSizqogKnOr2eBg_WSmXdqUej4w)

---

## 🧠 Project Motivation

Manual monitoring of mask compliance in public spaces is inefficient and prone to human error. This project demonstrates how **deep learning** and **computer vision** can automate the task — detecting faces and determining whether a person is wearing a mask via live webcam feed.

---

## 🔍 How It Works

- ✅ **Face Detection**: Uses Haar Cascade Classifier to detect faces in real time.
- ✅ **Mask Classification**: A trained CNN classifies each detected face as **Masked** or **Unmasked**.
- ✅ **Real-Time Inference**: Frame-by-frame prediction with confidence scores.
- ✅ **Visual Feedback**: Displays labeled bounding boxes around faces.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – For face detection and live video streaming
- **TensorFlow / Keras** – For deep learning model creation and training
- **NumPy** – For efficient numerical operations
- **Matplotlib / Seaborn** – For model visualization and evaluation

---

## 🚀 Features

- 📹 Live webcam-based face mask detection
- 🔍 Real-time classification with bounding boxes
- 📊 Confidence score display for predictions
- 💡 Lightweight and fast enough for edge devices

---

## 🧠 Model Architecture

The model follows a simple yet effective CNN structure:

- 3 × Conv2D layers with ReLU and MaxPooling
- Dropout for regularization
- Flatten + Dense layers
- Final layer with **sigmoid** or **softmax** activation (depending on binary/multiclass setup)

> 🎯 **Accuracy**: Achieved ~98% validation accuracy on custom mask detection dataset
