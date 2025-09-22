# IR Gesture Recognition using TinyML (ESP32 + Python)

A **real-time gesture classification system** built with an **IR sensor** and **ESP32**. This project detects **three types of human gestures** — *pass*, *stay*, and *wave* — using a lightweight **Random Forest** model running on the microcontroller.

---

## 📘 Problem Statement

Detecting gesture types like passing, pausing, or waving is challenging using simple analog sensors. Traditional methods either require **complex camera systems** or **large ML models**, which are not feasible for low-power microcontrollers.

**Goal:** Build a **compact, accurate, and real-time gesture recognition system** using:
- Simple IR analog sensor
- Lightweight ML model (Random Forest)
- Real-time inference on ESP32

---

## 🛠️ Hardware & Connections

**Microcontroller:** ESP32 Dev Module  
**Sensor:** IR analog sensor (e.g., TSOP)  

**Connections:**
- Sensor OUT → GPIO34 (analog input)
- GND → GND
- VCC → 3.3V
- USB → Serial (9600 baud)

---

## 📊 Data Collection

- Collected **30 analog samples** every 100 ms (3-second window)
- Labeled 3 gestures:
  - **pass** – quick motion past sensor
  - **stay** – steady pause in front of sensor
  - **wave** – alternating high/low signals
- Stored in `gestures.csv`
- Each row: `label, f1, f2, ..., f30`
- Values normalized to **[0, 1]**

---

## 💻 Model Training & Export

- **Training script:** `train_gesture_model.py`
- **Library:** EverywareML
- **Algorithm:** RandomForestClassifier (10 trees)
- **Dataset split:** 80% train / 20% test
- **Export:** `model.h` for ESP32

**Performance:**  
- Training Accuracy: ~90.91%  
- Model runs directly on ESP32 as C++ code — no TFLite required

---

## 🧠 Local Inference (Python + TFLite)

- **Script:** `predict_ir_gesture.py`
- **Model:** `ir_gesture_model.tflite`
- ESP32 sends **30 IR readings via Serial** (COMx, 115200 baud)
- Python script receives data, runs inference, and prints predictions:

