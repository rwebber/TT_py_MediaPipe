# MediaPipe Pose Tracking for Isadora – Pythoner Integration
A complete example project demonstrating how to integrate **MediaPipe Pose** tracking into **Isadora** using the built-in **Pythoner actor**.  
This setup provides real-time full-body motion tracking using only a standard webcam—no depth camera or external hardware required.

---

## 🌟 **Overview**
This example shows how to stream live pose-tracking data from MediaPipe directly into Isadora, where it can be used for:

- Interactive installations  
- Live VJ and performance visuals  
- Motion-controlled lighting or audio  
- Dance/movement analysis  
- Rapid prototyping for gesture-driven interfaces  

The project includes tools for both **raw 2D landmark tracking** and **conversion into Isadora’s native skeleton format**, enabling immediate use with existing actors and patches.

---

## 📦 **What’s Included**

### ✔ Pythoner-Based Pose Tracker
- Captures all **33 MediaPipe Pose landmarks**  
- Uses OpenCV to read from your webcam  
- Outputs structured landmark data into Isadora  
- Runs in real time on macOS and Windows  

### ✔ OpenNI-Compatible Conversion Tools
MediaPipe and OpenNI use different joint sets.  
This project includes automatic conversion to:

- 15-joint OpenNI skeleton  
- Proper Isadora bone names  
- Smoothed and normalized coordinates  

### ✔ Skeleton JSON Builder
Produces JSON in the correct structure for:

- **Skeleton From JSON** (Isadora actor)  
- Custom rigging workflows  
- Interoperability with other tracking systems  

### ✔ Automatic Environment Setup
A lightweight tool is included to:

- Create a virtual environment for Pythoner  
- Install all required Python libraries  
  - `mediapipe`  
  - `opencv-python`  
  - `numpy`  
- Ensure consistency across macOS and Windows  

---

## 🚀 **Getting Started**

### 1. Download or clone this repository
```
git clone https://github.com/<yourname>/Isadora-MediaPipe-Pose.git
```

### 2. Keep all included files together
The Pythoner system depends on relative paths.  
Do not move the `.izz` file or the `/python_modules` folder independently.

### 3. Open the Isadora project
Load the `.izz` file included in this repository.

### 4. When prompted, allow Pythoner to run
If this is your first time using Pythoner on the system, Isadora may ask for permission.

### 5. Enter Setup Scene and click the Setup Trigger
This will:

- build a virtual environment  
- install dependencies  
- configure Python paths  

### 6. Restart Isadora
Required once the environment has been created.

### 7. Begin tracking!
Your webcam feed will be analyzed live, and pose/skull/spine joint data will flow into Isadora.

---

## 🦴 **Skeleton Integration Workflow**

Once MediaPipe landmarks are captured:

### Step 1 — Convert to OpenNI format  
The included mapper condenses MediaPipe’s 33-point model into a 15-joint OpenNI-style skeleton.

### Step 2 — Package into Isadora JSON  
The JSON wrapper outputs an exact match for:

```
Skeleton From JSON  →  Skeleton Visualizer / Skeleton Mapper
```

### Step 3 — Play  
Create interactive projects in Isadora using Pose tracking in real-time.


---

## 💡 **Why Use This?**

- Free, open-source, and cross-platform  
- Works with any standard webcam  
- No external trackers or depth sensors required  
- Simple, modular, and expandable  
- Plug-and-play integration with Isadora’s native skeleton ecosystem  

If you want to add body tracking to an installation quickly, this is a great choice.

---

## 📁 **Repository Structure**
```
Isadora-MediaPipe-Pose/
│
├── Isadora-Mediapipe-Demo.izz      # Example project
├── python_modules/                 # Python code for Pythoner
│   ├── mediapipe_pose-detection.py
│   └── ...
├── virtual_env/                    # NOT PART OF THE DOWNLOAD - Created later by the Setup Scene
└── README.md                       # You are here
```

---

## 🖥 **Requirements**

### Isadora
- Version 4 or later  
- Pythoner actor installed (included with Isadora)

### Operating System
- macOS (Intel or Apple Silicon)  
- Windows 10+  

### Hardware
- Any USB webcam  
- (Optional) GPU for improved pose-tracking performance  

---

## 🔧 **Troubleshooting**

### ⚠ Pythoner Errors About Missing Packages
Re-run the **Setup** trigger to reinstall dependencies.

### ⚠ Skeleton Looks Distorted
Ensure the JSON being sent to *Skeleton From JSON* matches the correct structure.

---

## 🤝 **Contributing**
PRs and feature suggestions are welcome — especially improvements to the skeleton mapping logic or examples showing creative uses inside Isadora.

---

## 📜 **License**
MIT License  
You may use this example freely in commercial and non-commercial projects.

---

## 🙏 **Credits**
Developed by **Ryan Webber (DusX)**  
Part of ongoing tooling and workflow improvements for **TroikaTronix Isadora**.
