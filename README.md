# KnightSight ANPR — Edge-Optimized License Plate Recognition

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A two-stage YOLO + ONNX-RapidOCR pipeline that detects vehicles, localizes their plates, and reads them in <250 ms on CPU, with skew correction and two-line plate support — all under 32 MB and 2 GFLOPs.

---

## Demo

*(Placeholder for GIF/screenshots)*

---

## Table of Contents
1. [Features](#features)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Performance & Benchmarks](#performance--benchmarks)
4. [Tools & Tech Stack](#tools--tech-stack)
5. [Quick Start](#quick-start)
6. [Training Your Own Plate Detector](#training-your-own-plate-detector)
7. [How It Works](#how-it-works)
8. [Project Structure](#project-structure)
9. [Configuration & Tunables](#configuration--tunables)
10. [Limitations & Future Work](#limitations--future-work)
11. [Acknowledgments](#acknowledgments)
12. [License](#license)

---

## Features

- **Two-Stage Cascade Detection**: Vehicle tracking followed by localized plate ROI extraction significantly reduces compute overhead on irrelevant pixels.
- **ONNX Runtime Export**: YOLO models are automatically exported to ONNX, delivering 3–4× faster inference times on CPU compared to native PyTorch.
- **RapidOCR Integration**: Utilizes PaddleOCR weights without the heavy PaddlePaddle dependency, featuring a blistering-fast recognition-only (rec-only) path.
- **Aspect-Ratio-Aware Routing**: Intelligently routes multi-line plates (e.g., motorcycle or square plates) based on bounding box proportions.
- **Strict Hough-Line Deskew**: Applies multi-guard rejection (minimum lines, angle standard deviation, ±12° cap) to prevent over-rotation on false edges like vehicle bumpers.
- **Replicate-Border Padding**: 6% padding preserves edge characters during affine warping and deskewing operations.
- **Multi-Format Regex Validation**: Robust validation covering 7 plate formats (including Indian short/full, Vietnamese, European, US, and generic).
- **Garbage Rejection**: Strict length-bounded acceptance (4–11 characters) prevents confidently-wrong "NO_READ" scenarios when the OCR misfires on tilted or obscured plates.
- **Multi-Variant Robustness**: Built to degrade gracefully across challenging environmental conditions like dim lighting, night scenes, motion blur, and sensor noise.
- **Cold-Start Warm-Up**: Pre-warms the OCR and ONNX detectors using 5 dummy inference passes to ensure stable baseline latency.
- **Dual Deployment**: Ready for both single-image analysis and streaming video processing with intelligent frame-skipping.

---

## Pipeline Architecture

```text
[Input Image/Video Frame]
         │
         ▼
 1. Vehicle Detection (YOLO11n, conf=0.15)
         │
         ├──► (Cropped Vehicle ROI)
         │
         ▼
 2. Plate Detection (YOLO11s custom, conf=0.10)
         │
         ├──► (Cropped Plate ROI)
         │
         ▼
 3. Pre-processing (6% Replicate Padding + Hough Deskew)
         │
         ├──► Is Aspect Ratio (h/w) > 0.40?
         │
   ┌─────┴─────┐
[YES]        [NO]
   │           │
   ▼           ▼
4. Two-Line  4. Single-Line
   Split       Path
   │           │
   ▼           ▼
5. Prep + CLAHE L-Channel
   │           │
   ▼           ▼
6. Rec-Only Fast Path (RapidOCR)
   │           │
   └──► Fails? ──► Full Det+Rec Fallback
         │
         ▼
 7. Regex Format Validation & Length Checks
         │
         ▼
[Annotated Output + JSON Report]
```

1. **Vehicle Detection** — YOLO11n (COCO classes 2/3/5/7: car, motorcycle, bus, truck), exported to ONNX, `IMG_SIZE=320`, `conf=0.15`
2. **Vehicle ROI Crop** — Isolates the vehicle to discard irrelevant background noise.
3. **Plate Detection** — Custom-trained YOLO11s (`best.pt`) on a Vietnamese parking-lot dataset (6,170 images), exported to ONNX, `IMG_SIZE=320`, `conf=0.10`.
4. **Plate Deskew** — 6% replicate-padding is applied followed by Hough-based skew correction (`deskew_plate`).
5. **Aspect-Ratio Routing** — If `h/w > 0.40`, the system routes to the two-line plate logic; otherwise, it proceeds as a single-line plate.
6. **Single-Line OCR** — The plate is prepared (`prep_plate`: 4%/2% inset + 56-px upscale + CLAHE on the L-channel) and passed to the RapidOCR rec-only fast path. A full detection+recognition fallback is triggered upon failure.
7. **Two-Line OCR** — The plate undergoes a horizontal half-split. Each half runs the rec-only path, with a full det+rec fallback that vertically groups rows if necessary.
8. **Format Validation** — Results are strictly checked against regex definitions (Indian, Vietnamese, European, US, generic) with garbage rejection (length 4–11 bounds).
9. **Output Generation** — The pipeline yields the annotated image/video (green vehicle box, red plate box, plate text), structured JSON, and a rubric report.

---

## Performance & Benchmarks

Benchmarked on Google Colab CPU (Intel Xeon @ 2.20GHz) with `IMG_SIZE=320`.

| Metric | Achieved | Target | Status |
| :--- | :--- | :--- | :---: |
| **FLOPs total** | 1.85 GFLOPs | ≤ 5.0 GFLOPs | PASS |
| **Latency** | ~200 ms | ≤ 250 ms | PASS |
| **Model Size** | 31.7 MB | ≤ 150 MB | PASS |
| **Vehicle mAP@0.5** | 0.547 | ≥ 0.50 | PASS |
| **Plate mAP@0.5** | 0.950 | ≥ 0.85 | PASS |
| **OCR Character Accuracy** | 0.900 | ≥ 0.80 | PASS |
| **Robustness Retention** | 0.75+ | ≥ 0.70 | PASS |

*(Robustness retention measured across dim, night, blur, and noise variants using Levenshtein-based CER for OCR accuracy).*

---

## Tools & Tech Stack

- **Python** 3.12
- **PyTorch** 2.10
- **Ultralytics** 8.4.41 (YOLO11)
- **ONNX** 1.21 + **ONNX Runtime** 1.25 + **onnxsim** 0.1.91
- **RapidOCR** 1.x (`rapidocr-onnxruntime`)
- **OpenCV** 4.x
- **NumPy** >=2.1,<2.3
- **thop** (FLOPs profiler)
- **Environment**: Google Colab (primary) + local Windows venv

---

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/KnightSight-ANPR.git
   cd KnightSight-ANPR
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -q ultralytics opencv-python "numpy>=2.1,<2.3" thop rapidocr-onnxruntime onnx onnxruntime
   ```

3. **Provide the weights**:
   Place your custom trained `best.pt` file at `/content/best.pt` (if on Colab) or inside the `weights/` directory (if running locally).

4. **Launch the Image Pipeline**:
   Open `notebooks/anpr_image.ipynb` in Google Colab or your local Jupyter environment.

5. **Upload the model**:
   Run **Cell A** to upload `best.pt` into the environment.

6. **Initialize the models**:
   Run **Cell B**. This will automatically install dependencies, export the YOLO PyTorch models to highly-optimized ONNX format, and run the warm-up passes.

7. **Run Inference**:
   At the upload prompt, drag and drop a test image. The pipeline will process it and print the annotated output, JSON payload, and rubric metrics inline.

8. **Process Video**:
   Open `notebooks/anpr_video.ipynb`. Upload an `.mp4` file and the pipeline will output a downloadable annotated video file.

9. **Tune Performance**:
   Modify `IMG_SIZE` within the notebook cells to trade off accuracy vs. latency (`320` is default; drop to `224` for sub-150ms latency, or increase to `480` for capturing distant plates).

---

## Training Your Own Plate Detector

The plate detection model (`best.pt`) was custom-trained specifically for this pipeline.

- **Dataset**: 6,170 images sourced from Vietnamese parking lots. Classes were relabeled into a single `license_plate` class.
- **Offline Augmentation**: The dataset was heavily augmented to ensure environmental robustness. We generated 4 variants (night, blur, glare, occlusion) yielding 3,000 extra images each using `Albumentations`.
- **Training Setup**: Trained on YOLO11s with `imgsz=640` for 50 epochs using the AdamW optimizer.
- **Result**: Final plate detection mAP@0.5 reached **0.95**.

---

## How It Works

### Aspect-Ratio Routing
Standard recognition-only OCR models struggle natively with stacked text. To combat this, an aspect-ratio gate routes square crops (typically motorcycle plates) to a manual horizontal half-split path. Each half is processed individually. This completely bypasses the canonical "one-shot OCR" failure rate, especially on tightly-packed Indian motorcycle plates.
```python
h, w = plate_img.shape[:2]
if (h / max(w, 1)) > 0.40:
    split_text = _read_two_line(plate_img)
```

### Strict Hough-Line Deskew
Naive deskewing algorithms often lock onto the edges of a vehicle bumper instead of the plate, causing catastrophic over-rotation. We implemented strict guards (requiring ≥3 consistent lines, a standard deviation < 5°, and a hard ±12° cap) to guarantee that only genuine plate warping is corrected.

### Garbage Rejection
A critical flaw in many ANPR systems is confidently returning a hallucinated read on blurry crops. Our length-bounded garbage rejection ensures that if the OCR reads `1815073POCCO`, but the format regex expects a standard plate, the system gracefully degrades and returns `NO_READ` instead of polluting the database.

### Fast Path vs. Fallback
Clean plates are routed through a hyper-fast recognition-only path (~40 ms). The heavier, full detection+recognition fallback is only triggered if the fast path fails the strict regex validation, ensuring compute resources are preserved for 90% of standard reads.

---

## Project Structure

```text
├── README.md
├── requirements.txt
├── notebooks/
│   ├── anpr_image.ipynb
│   └── anpr_video.ipynb
├── weights/
│   └── best.pt
├── src/
│   ├── pipeline.py
│   └── train_plate.py
├── samples/
└── outputs/
```

---

## Configuration & Tunables

- `IMG_SIZE`: Adjusts the YOLO input tensor. Default is `320`. Lowering to `224` increases speed; raising to `480` increases long-range accuracy.
- `frame_skip`: (Video pipeline) Drops alternating frames (e.g., `frame_skip=2`) to double processing speed without introducing visible jitter.
- `conf`: Detection thresholds. Defaults are `0.15` for vehicles and `0.10` for plates to prioritize recall before OCR validation.

---

## Limitations & Future Work

- **Resolution Dependency**: OCR recognition drops significantly for plates smaller than 30 pixels in height.
- **Character Sets**: Currently optimized for Latin characters and numbers. Non-Latin scripts are not supported by the default RapidOCR weights.
- **Extreme Angles**: The deskew algorithm caps out at a 12° tilt to prevent false positives; extreme perspective angles will likely fail OCR.

---

## Acknowledgments

- Built for the **KnightSight EdgeVision Challenge**.
- Core object detection powered by [Ultralytics](https://ultralytics.com/).
- OCR pipeline leverages [RapidOCR](https://github.com/RapidAI/RapidOCR) and PaddleOCR architecture.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with precision. Deployed for security. Welcome to KnightSight.*
