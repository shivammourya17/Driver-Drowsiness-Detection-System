Driver Drowsiness Detection System

A real-time driver drowsiness detection system built using computer vision and machine learning techniques.
The system monitors a driver’s face through a webcam, detects signs of drowsiness (such as closed eyes or yawning), and triggers an alarm notification to prevent accidents.

🚨 Motivation

Drowsy driving is a major safety risk worldwide.
According to the National Highway Traffic Safety Administration (NHTSA):

Every year, 100,000+ police-reported crashes involve drowsy driving.

These crashes lead to over 1,550 fatalities and 71,000 injuries annually.

Since drowsiness can be difficult to measure at crash scenes, the true numbers may be even higher.
This project was built with the goal of saving lives by providing an early alert system for drowsy drivers.

⚙️ Features

Real-time face detection using OpenCV and Dlib.

Eye closure detection for identifying drowsiness.

Yawning detection for additional safety.

Instant audio alert (alarm sound) when drowsiness is detected.

Lightweight and efficient — runs on a standard laptop webcam.

🛠️ Tech Stack

OpenCV: Industry-standard library for real-time computer vision.

imutils: Helper utilities for working with OpenCV.

Dlib: Provides face landmark detection and ML-based CV algorithms.

scikit-learn: Machine learning library with easy-to-use API.

NumPy: Fundamental library for numerical computing in Python.

🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

1️⃣ Clone the Repository
```sh
git clone https://github.com/shivammourya17/Driver-Drowsiness-Detection-System.git
 ```


2️⃣ Install Dependencies

Install all required libraries:
```sh
pip install -r requirements.txt
 ```

3️⃣ Install Dlib

Windows: Install CMake and restart your terminal before installing dlib.

Linux/macOS: If you face issues with dlib, follow this guide.

4️⃣ Run the Application
```sh
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
 ```

## 🌟 Contributing

Feel free to contribute by creating pull requests or opening issues.
