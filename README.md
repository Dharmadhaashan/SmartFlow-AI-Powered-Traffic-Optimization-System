# 🚦 SmartFlow – AI Powered Traffic Optimization System

SmartFlow is an **AI-based intelligent traffic management system** designed to reduce urban congestion by dynamically controlling traffic signals using **Computer Vision and Audio AI**.

Traditional traffic signals operate using fixed timers, which cannot adapt to changing traffic conditions. SmartFlow solves this by analyzing **vehicle density using camera feeds** and detecting **emergency vehicle sirens using audio analysis**.

The system combines **Vision AI (traffic monitoring)** and **Sound AI (siren detection)** to optimize traffic flow and ensure faster passage for emergency vehicles.

---

# 🧠 System Architecture

SmartFlow uses two AI subsystems working together.

### 🚗 Vision AI – Traffic Density Detection

The Vision AI model processes traffic camera feeds using **OpenCV**.

It performs:

* Vehicle detection
* Vehicle counting
* Traffic density estimation

The system then dynamically adjusts traffic signal timings based on the number of vehicles detected in each lane.

---

### 🚑 Sound AI – Emergency Siren Detection

The Sound AI model detects emergency vehicle sirens using **audio classification**.

Audio signals are processed using **MFCC (Mel Frequency Cepstral Coefficients)** extracted from `.wav` audio files.

A **Convolutional Neural Network (CNN)** built using **TensorFlow/Keras** is trained to classify sounds into four categories:

* Ambulance
* Firetruck
* Police
* Traffic noise

When an emergency siren is detected, the system automatically prioritizes that traffic lane.

---

# 🔊 Sound AI Model Details

### Feature Extraction

Audio files are processed using **Librosa** to extract MFCC features.

Key parameters:

* MFCC Features: **40**
* Time steps: **130**
* Input Shape: **(40 × 130 × 1)**

These MFCC spectrograms are used as input to the CNN model.

---

### CNN Architecture

The Sound AI model uses the following architecture:

1. **Conv2D (32 filters)**
2. **MaxPooling2D**
3. **Dropout (0.2)**
4. **Conv2D (64 filters)**
5. **MaxPooling2D**
6. **Dropout (0.3)**
7. **Flatten Layer**
8. **Dense Layer (128 neurons)**
9. **Output Layer (Softmax)**

Loss Function:

```
categorical_crossentropy
```

Optimizer:

```
Adam
```

---

### Model Training

The dataset is automatically split:

* **80% Training**
* **20% Testing**

Training parameters:

* Epochs: **30**
* Batch Size: **32**

The model outputs:

* Testing Accuracy
* Classification Report
* Confusion Matrix

After training, the model is saved as:

```
siren_detector.h5
classes.npy
```

---

# 📂 Project Structure

```
SmartFlow-AI-Powered-Traffic-Optimization-System
│
├── sound ai/                # Current Sound AI model (siren detection)
│
├── traffic ai 2/            # Vision AI model using OpenCV for vehicle detection
│
├── traffic ai 1/            # Previous version of Sound AI model
│
├── road.pdf                 # Project documentation
│
└── README.md
```

---

# ⚙️ How the System Works

### Step 1 — Traffic Monitoring

Traffic cameras capture real-time footage of road intersections.

### Step 2 — Vehicle Detection

Vision AI analyzes frames using OpenCV to:

* Detect vehicles
* Count cars
* Estimate congestion levels.

### Step 3 — Traffic Signal Optimization

Traffic signals dynamically change based on vehicle density.

### Step 4 — Siren Detection

Audio input is processed by the Sound AI model:

1. Audio → MFCC Features
2. MFCC → CNN Model
3. CNN → Siren classification

### Step 5 — Emergency Priority

If a siren is detected:

* Traffic lights prioritize that lane
* Other lanes temporarily stop
* Emergency vehicles pass quickly.

---

# 🛠 Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* Librosa
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# 📊 Applications

* Smart City Infrastructure
* Intelligent Transportation Systems
* Emergency Vehicle Priority Systems
* Urban Traffic Optimization

---

# 🔮 Future Improvements

* Real-time **multi-intersection coordination**
* **Edge AI deployment on smart cameras**
* Reinforcement Learning for traffic optimization
* Integration with **IoT traffic sensors**
* Live city traffic monitoring dashboard

---

# 👨‍💻 Author

**Dharmadhaashan**

AI | Computer Vision | Machine Learning

---

# 📜 License

This project is open-source and available under the **MIT License**.
