# Deep Learning — PR-3
### Classical Image Processing & Deep-Learning-Based Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

A complete, end-to-end computer vision pipeline spanning classical image-processing
techniques and modern deep-learning-based detection models — built as Project 3
(Deep Learning) for a Computer Vision course.

The notebook moves progressively from low-level pixel operations to a combined,
real-time, multi-model detection system.

---

## 📋 Project Overview

| Task | Topic | Key Techniques |
|---|---|---|
| **1** | Environment Setup & Morphology | Thresholding, erosion, dilation, opening, closing, kernel shape/size comparison |
| **2** | Bitwise Operations & Histograms | AND / OR / XOR / NOT masking, ROI extraction, grayscale & colour histograms, brightness/contrast |
| **3** | Face Detection (YuNet) | `cv2.FaceDetectorYN`, 5-point landmark detection, real-time webcam inference, privacy blurring |
| **4** | Object Detection (YOLOv8) | Pretrained COCO detection, confidence/IoU tuning, per-class analysis |
| **5** | Integrated Pipeline | Combined face + object detection, morphological pre-cleaning, FPS benchmarking |

---

## 🎥 Demo

> Add a link/embed to your demo video here once recorded, e.g.:
> `[Watch the demo video](your-video-link-here)`

---

## 🗂️ Repository Structure

```
PR-3/
├── data/
│   ├── images/
│   │   ├── lena.jpg
│   │   ├── baboon.jpg
│   │   ├── butterfly.jpg
│   │   ├── cards.png
│   │   ├── home.jpg
│   │   ├── leuvenA.jpg
│   │   ├── messi5.jpg
│   │   └── (face images excluded — see Privacy Note below)
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
├── yolov8n.pt                  # auto-downloaded on first run
├── CV-PR-3.ipynb               # main notebook — all 5 tasks
├── CV-PR-3.html                # exported HTML version (rendered outputs)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd PR-3
```

**2. Install dependencies**
```bash
pip install opencv-python opencv-contrib-python ultralytics matplotlib numpy
```

**3. Download required assets**

| Asset | Source | Destination |
|---|---|---|
| Sample images | [OpenCV samples repo](https://github.com/opencv/opencv/tree/master/samples/data) | `data/images/` |
| YuNet weights | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) → `face_detection_yunet_2023mar.onnx` | `data/models/` |
| YOLOv8-nano weights | Auto-downloaded via `YOLO("yolov8n.pt")` on first run | project root |

**4. Launch the notebook**
```bash
jupyter notebook CV-PR-3.ipynb
```
or open it directly in VS Code.

---

## 🧠 Models Used

- **YuNet** — a compact, millisecond-level CNN face detector shipped with OpenCV
  5.0 via `cv2.FaceDetectorYN`. Returns bounding boxes, 5-point facial landmarks,
  and a confidence score in a single forward pass.
  → [OpenCV Zoo: face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

- **YOLOv8-nano** — the smallest variant of Ultralytics' YOLOv8, pretrained on
  the 80-class COCO dataset. Chosen for its speed/accuracy balance suited to
  real-time inference on modest hardware.
  → [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📊 Results Summary

| Technique | Type | What It Detects | Lighting Robustness | Approx. FPS | Best Use Case |
|---|---|---|---|---|---|
| Morphology (erode/dilate/open/close) | Classical | Noise/shape cleanup in binary masks | Low (depends on threshold) | Very High (>100) | Pre-cleaning masks before detection |
| Bitwise Ops & Histograms | Classical | Masking, brightness/contrast diagnostics | Low–Medium | Very High (>100) | ROI extraction, exposure diagnostics |
| YuNet Face Detection | Deep Learning | Faces + 5 landmarks | High | Medium (hardware-dependent) | Face-focused apps (attendance, blur/privacy) |
| YOLOv8-nano | Deep Learning | 80 COCO object classes | High | Medium (hardware-dependent) | General object detection, surveillance |

*(See Section 5.4 in the notebook for full measured FPS numbers on the test machine.)*

---

## 🔍 Key Findings

- Classical operations (morphology, bitwise logic, histograms) run extremely
  fast and are best used as lightweight **pre-processing** steps ahead of a
  deep model, not as replacements for it.
- Deep-learning detectors generalize far better across lighting, pose, and
  occlusion than classical approaches, at the cost of higher compute and
  lower FPS.
- Threshold parameters (`score_threshold` for YuNet; `conf` / `iou` for YOLO)
  should be tuned to the deployment context — lower thresholds for
  safety-critical real-time monitoring, higher thresholds where false
  positives are more costly than occasional misses.
- Combining multiple models into one pipeline is straightforward but carries
  a measurable FPS cost that should be benchmarked against target hardware
  before deployment.

---

## 🔒 Privacy Note

Personal face photographs used for Task 3 (face detection) are **excluded**
from this repository (`.gitignore`) to protect the privacy of the individuals
photographed. As a result, Task 3 cells will not execute end-to-end on a fresh
clone without supplying your own face images in `data/images/`. The rendered
outputs are preserved in `CV-PR-3.html` and within the notebook's saved cell
outputs.

---

## 🛠️ Tech Stack

- Python 3.12
- OpenCV 5.0 (`cv2.FaceDetectorYN`, morphology, bitwise ops, histograms)
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Jupyter Notebook / VS Code

---

## 📚 References

- Z. Zhang et al., *YuNet: A Tiny Millisecond-level Face Detector*, [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- G. Jocher et al., *Ultralytics YOLOv8*, [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Sample Data Repository](https://github.com/opencv/opencv/tree/master/samples/data)

---

## 👤 Author

**Name:** Maitrak kunjadiya
**Course:** AI&ML [Computer Vision] — PR-3 (Deep Learning)
**Date:** 10-09-2026

---

## 📄 License

This project is submitted for academic coursework purposes.
