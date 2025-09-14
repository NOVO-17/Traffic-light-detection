# 🚦 Traffic Light Detection using OpenCV & HSV

This project implements a **real-time traffic light detection system** using **OpenCV** and the **HSV color space**.

It can detect **Red**, **Yellow**, and **Green** lights from a webcam feed, video, or still images.


## 📌 Features

* Detects **Red, Yellow, Green** traffic lights using HSV color thresholds.
* Works with **live webcam feed**, **video files**, or **static images**.
* Uses **morphological operations** to reduce noise.
* Displays bounding boxes and labels on detected lights.
* Prints detection status (**STOP / READY / GO**) in console.


## 🛠️ Installation

### Requirements

* Python 3.7+
* OpenCV
* NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```


## ▶️ Usage

### 1. Run with Webcam

```python
image = cv2.VideoCapture(0)
```

### 2. Run with Image
Change code to load image:

```python
image = cv2.imread("traffic_light.jpg")
```

### 3. Run with Video
Replace video capture:

```python
cap = cv2.VideoCapture("traffic_light_video.mp4")
```


## 🔑 HSV Color Ranges

The system uses HSV ranges for color segmentation:

```python
# Red
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Yellow (finely tuned)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Green
lower_green = np.array([40, 100, 100])
upper_green = np.array([90, 255, 255])
```

> ⚠️ These values may need tuning depending on **lighting conditions** and **camera type**.


## 📸 Example Output

* **RED** → "STOP"
* **YELLOW** → "READY"
* **GREEN** → "GO"

Each detection is shown with a **bounding box + label**.

## 📜 License

MIT License – Free to use for education, research, and projects.
