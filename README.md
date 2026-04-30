# Weld Defect Detection System
### AI-Powered Portable Robotic Inspection using YOLOv8 + Raspberry Pi 4B

A real-time weld defect detection system trained on a custom dataset and deployed on a Raspberry Pi 4B mounted on a 6-axis robotic arm for inspection of difficult-to-access weld joints.

---

## Project Overview

This system uses a YOLOv8 Nano model to detect two critical weld defect categories in real time:

| Class | Code | Description |
|-------|------|-------------|
| Crack | CR | Linear fractures in weld metal or heat-affected zone |
| Incomplete Penetration | IP | Weld metal failing to fully penetrate the joint root |

---

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 (Overall) | 74.5% |
| mAP@50-95 | 51.5% |
| Precision | 71.2% |
| Recall | 76.0% |
| Inference Speed | 1.9ms per image |
| Processing Speed | ~526 FPS on RTX 4050 |
| Model Parameters | ~3 Million (YOLOv8 Nano) |

---

## Dataset

| Property | Value |
|----------|-------|
| Training Images | 1,610 |
| Validation Images | 561 |
| Defect Instances | 529 |
| Background Images | 299 |
| Classes | 2 (CR, IP) |

---

## System Architecture

```
6-Axis Robotic Arm
│
▼
5MP Raspberry Pi Camera Module (Rev 1.3)
│
▼
Raspberry Pi 4B (Edge Computing)
Running NCNN YOLOv8 Model
│
▼
Real-time Detection Output
Micro HDMI Display + Alert System
```

---

## Project Structure

```
PROJECT/code/
│
├── detect_weld.py        # Main detection script (webcam/image/batch)
├── train_weld.py         # Model training script
├── val.py                # Model evaluation and metrics
├── split_dataset.py      # Dataset train/val split utility
├── remap_labels.py       # Dataset class remapping utility
├── export_model.py       # Export to NCNN for Raspberry Pi
├── pi_detect.py          # Detection script for Raspberry Pi 5
│
├── Weld_dataset/         # Dataset (not included in repo — see below)
│   ├── images/
│   │   ├── train/        # 1,610 training images
│   │   └── val/          # 561 validation images
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
└── runs/                 # Training outputs (not included in repo)
    └── detect/
        └── train_weld_4class4/
            └── weights/
                └── best.pt   # Trained model weights
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/weld-defect-detection.git
cd weld-defect-detection

# Install dependencies
pip install ultralytics opencv-python
```

---

## Usage

### Run Detection (Webcam / Image / Batch)
```bash
python detect_weld.py
```
Select from:
- `1` — Webcam real-time detection
- `2` — Single image (file picker opens)
- `3` — Batch folder (folder picker opens)

### Train the Model
```bash
python train_weld.py
```

### Evaluate Performance
```bash
python val.py
```

### Export for Raspberry Pi
```bash
python export_model.py
```

---

## Raspberry Pi Deployment

The model is exported to NCNN format for deployment on Raspberry Pi 4B without GPU dependency. Expected performance: 8–15 FPS on ARM Cortex-A72.

```bash
# On Raspberry Pi 4B
pip install ultralytics
python pi_detect.py
```

---

## Hardware Requirements

### Development (Training)
- NVIDIA GPU (RTX 4050 used)
- Python 3.8+
- 8GB RAM minimum

### Deployment

| Component | Details |
|-----------|---------|
| Microcomputer | Raspberry Pi 4 Model B (2GB RAM) |
| Camera | 5MP Raspberry Pi Camera Module for Pi 3/4 Model B — Rev 1.3 |
| Display Cable | Official Micro HDMI Male to Standard HDMI Male Cable |
| Power Supply | Official USB Type-C 15.3W Power Supply for Raspberry Pi 4 (Black) |
| Cooling | Raspberry Pi Black 4-in-1 Aluminum Heat Sink Set for Pi 4B |
| Enclosure | Official Raspberry Pi 4 Case |
| Robotic Arm | 6-axis robotic arm (for industrial deployment) |

---

## Results

Detection examples and training curves are available in the `runs/` folder after training. Key output files:

- `results.png` — Loss and mAP curves over 150 epochs
- `confusion_matrix_normalized.png` — Per-class accuracy
- `val_batch0_pred.jpg` — Sample predictions on validation images

---

## Project Context

| Field | Detail |
|-------|--------|
| Institution | Ajay Kumar Garg Engineering College, Ghaziabad |
| Programme | B.Tech Mechanical Engineering |
| Academic Year | 2025–2026 |
| Supervisor | Ms. Gaganpreet Kaur |

---

## License

For academic use only.