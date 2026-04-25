# 🛡️ KnightSight EdgeVision ALPR

![KnightSight Logo](https://img.shields.io/badge/KnightSight-EdgeVision-00ffcc?style=for-the-badge&logo=shield&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-yellow?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**KnightSight EdgeVision** is a blazing-fast, high-accuracy Automatic License Plate Recognition (ALPR) dashboard. It combines the power of **YOLO object detection** with **RapidOCR** and wraps it all in a premium, cinematic, edge-computing-inspired UI.

---

## ✨ Features

- **🚀 Auto-ONNX Optimization**: Automatically exports PyTorch `.pt` models to `.onnx` and optimizes CPU multi-threading for near real-time inference without needing a GPU.
- **🚗 Multi-Vehicle Detection**: Accurately detects Cars, Motorcycles, Buses, and Trucks using YOLO.
- **🔡 Advanced Plate Reading**: Smart detection logic automatically identifies single-line vs. two-line plates, handling extreme aspect ratios and performing robust character parsing.
- **🎨 Cinematic Dashboard**: A gorgeous HTML5 Canvas frontend injected dynamically into Streamlit, featuring glassmorphism, dynamic progress gauges, typewriter effects, and real-time JSON previews.

---

## 🛠️ Tech Stack

- **Backend Architecture**: Streamlit, Python
- **Computer Vision**: Ultralytics (YOLOv8/11), OpenCV
- **Optical Character Recognition**: RapidOCR (ONNX Runtime)
- **Frontend UI**: Vanilla JS, HTML5 Canvas, CSS3 Variables

---

## 📦 Installation & Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd knightsight
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install streamlit ultralytics rapidocr-onnxruntime opencv-python numpy
   ```

3. **Provide the Models**:
   Ensure the following YOLO weight files are in the root directory:
   - `yolo11n.pt` (Standard vehicle detection model)
   - `best.pt` (Custom trained license plate detection model)

---

## 🚀 Usage

To spin up the KnightSight EdgeVision dashboard, run:

```bash
python -m streamlit run app.py
```

The server will start locally (usually on `http://localhost:8501`).
Upload a `.jpg` or `.png` of a vehicle, and the backend will process the image, detect the vehicle and plate, run OCR, and visually inject the bounding boxes and confidence scores directly onto the futuristic UI canvas.

---

## 🧠 Pipeline Architecture

1. **Upload**: User submits image via Streamlit.
2. **Vehicle Detection**: YOLO scans the full frame for vehicles.
3. **Plate Localization**: YOLO scans the cropped vehicle (or falls back to the full frame) to isolate the license plate.
4. **Plate Pre-processing**: CLAHE contrast adjustments and deskewing via Hough Lines prepare the crop.
5. **OCR Pipeline**: RapidOCR reads the characters. Split logic handles stacked/two-line motorcycle plates.
6. **Data Formatting**: Inference results are packed into a strict JSON schema and passed to the JavaScript frontend for real-time visualization.

---

*Built with precision. Deployed for security. Welcome to KnightSight.*
