
````markdown
# Obstacle Avoiding Robot Using Arduino & Computer Vision Object Detection App

![Obstacle Avoiding Robot Using Arduino](Obstacle%20Avoiding%20Robot%20Using%20Arduino.jpg)

This project combines embedded robotics and AI-based computer vision to demonstrate two powerful applications of automation:

1. An Arduino-based autonomous obstacle-avoiding robot that navigates its surroundings using an ultrasonic sensor.
2. A YOLO Object Detection Streamlit App for real-time image and video object detection powered by TensorRT and GPU acceleration.

---

## Table of Contents

* [Introduction](#introduction)
* [Objectives](#objectives)
* [Hardware Components](#hardware-components)
* [Circuit Diagram & Connections](#circuit-diagram--connections)
* [Arduino Code](#arduino-code)
* [Code Explanation](#code-explanation)
* [Applications](#applications)
* [YOLO Object Detection Streamlit App](#yolo-object-detection-streamlit-app)
  * [Features](#features)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [How to Run](#how-to-run)
* [Output](#output)
* [Conclusion](#conclusion)
* [License](#license)

---

## Introduction

The Obstacle Avoiding Robot is an autonomous system that can detect and avoid obstacles without human intervention. Using an Arduino Uno, ultrasonic sensor (HC-SR04), and L298N motor driver, the robot intelligently changes direction when it detects nearby objects.

The project also includes an advanced YOLO-based Object Detection Streamlit application, showcasing the AI side of robotics — detecting objects from images and videos in real-time using GPU acceleration.

---

## Objectives

1. Build an autonomous robot capable of avoiding obstacles.  
2. Demonstrate the integration of sensors, actuators, and microcontrollers using Arduino.  
3. Provide hands-on experience in motor control, sensor interfacing, and AI-powered computer vision.

---

## Hardware Components

| Component                   | Description                            |
| --------------------------- | -------------------------------------- |
| Arduino Uno                 | Central processing unit for the robot. |
| Ultrasonic Sensor (HC-SR04) | Measures the distance to obstacles.    |
| L298N Motor Driver          | Controls motor direction and speed.    |
| 2 DC Motors                 | Drive the robot’s wheels.              |
| 9V Battery                  | Powers the Arduino Uno.                |
| AA Battery Pack             | Provides power to the motors.          |
| Connecting Wires            | Assemble the circuit.                  |

---

## Circuit Diagram & Connections

### Ultrasonic Sensor:

| Pin  | Connection |
| ---- | ---------- |
| VCC  | Arduino 5V |
| Trig | Pin 4      |
| Echo | Pin 2      |

### Motor Driver (L298N):

| Pin                  | Connection    |
| -------------------- | ------------- |
| ENA                  | Arduino Pin 5 |
| ENB                  | Arduino Pin 6 |
| Left Motor Forward   | Pin 8         |
| Left Motor Backward  | Pin 9         |
| Right Motor Forward  | Pin 10        |
| Right Motor Backward | Pin 11        |

---

## Arduino Code

```cpp
void setup() {
  pinMode(4,OUTPUT);   // Trigger
  pinMode(2,INPUT);    // Echo
  pinMode(5,OUTPUT);   // EnA
  pinMode(6,OUTPUT);   // EnB
  pinMode(8,OUTPUT);   // Left motors forward
  pinMode(9,OUTPUT);   // Left motors backward
  pinMode(10,OUTPUT);  // Right motors forward
  pinMode(11,OUTPUT);  // Right motors backward
  Serial.begin(115200);
}

void loop() {
  digitalWrite(4, LOW);
  delayMicroseconds(2);
  digitalWrite(4, HIGH);
  delayMicroseconds(10);
  digitalWrite(4, LOW);
  int duration = pulseIn(2, HIGH);
  int distance = duration * 0.034 / 2;
  Serial.println(distance);
  delay(10);

  if (distance >= 20) {
    analogWrite(5, 255);
    digitalWrite(8, 1);
    digitalWrite(9, 0);
    analogWrite(6, 255);
    digitalWrite(10, 0);
    digitalWrite(11, 1);
  } else if (distance < 20) {
    analogWrite(5, 255);
    digitalWrite(8, 1);
    digitalWrite(9, 0);
    analogWrite(6, 255);
    digitalWrite(10, 1);
    digitalWrite(11, 0);
  }
}
````

---

## Code Explanation

* **Initialization:** Sets up sensor and motor pins.
* **Distance Calculation:** Uses the ultrasonic sensor with the formula
  `distance = duration × 0.034 / 2`.
* **Decision Logic:**

  * If distance ≥ 20 cm → Move forward.
  * If distance < 20 cm → Turn to avoid the obstacle.
* **Motor Control:** The L298N motor driver uses PWM to control motor direction and speed.

---

## Applications

* Autonomous Vehicles: Core logic for collision avoidance.
* Robotics Education: Great beginner project for Arduino & robotics.
* Home Automation: Adaptable for robotic vacuum cleaners and lawn mowers.

---

## YOLO Object Detection Streamlit App

This app demonstrates real-time object detection using YOLO + TensorRT in Streamlit, with full GPU acceleration support.

### Features

* Detects objects in images and videos.
* Uses TensorRT engine for optimized inference speed.
* GPU memory and inference time monitoring.
* Download annotated output images/videos.
* Multi-threaded processing for large videos.

---

### Sample Code (YOLO + Streamlit + TensorRT)

```python
import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image

st.title("YOLO Object Detection App")

MODEL_PATH = "runs/detect/train/weights/best.engine"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH, task='detect')
    return model

model = load_model()

conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    results = model.predict(source=image, conf=conf_threshold, device=DEVICE, verbose=False)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects")
```

---

## Requirements

* Python 3.9+
* CUDA-enabled GPU
* Installed libraries:

  ```bash
  pip install streamlit ultralytics torch torchvision torchaudio pillow opencv-python
  ```

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Obstacle-Avoiding-Robot-Arduino-YOLO.git
   cd Obstacle-Avoiding-Robot-Arduino-YOLO
   ```

2. Set up the environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your YOLO TensorRT model path is correct inside `app.py`:

   ```python
   MODEL_PATH = "path/to/best.engine"
   ```

---

## How to Run

### Run the Arduino Robot

1. Assemble components as per the circuit diagram.
2. Upload Arduino code using the Arduino IDE.
3. Power the robot to start autonomous navigation.

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL in your browser.

---

## Output

* Robot autonomously avoids obstacles.
* YOLO app detects and annotates objects in real time.

---

## Conclusion

The project merges robotics and AI to build an autonomous robot capable of environmental awareness through sensors and computer vision — a step toward intelligent robotic systems.

---

## License

This project is open-source under the MIT License.
Feel free to use, modify, and share it with attribution.

```

---

هل تحب أضيف بعد الكود ده كمان قسم **“Demo Images”** زي اللي قبل، فيه الصورتين `1.jpg` و `2.jpg` جنب بعض في النص داخل README؟
```
