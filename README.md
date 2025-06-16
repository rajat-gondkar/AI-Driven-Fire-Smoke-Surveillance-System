# Fire and Smoke Detection System

A real-time fire and smoke detection system using YOLOv8 and Streamlit, designed to leverage existing CCTV infrastructure for rapid emergency response.

---

## Motivation

In many urban and industrial setups, fire detection systems rely on costly and complex hardware installations. This project aims to repurpose widely available CCTV camera systems to detect fire and smoke using AI, enabling faster detection without the need for additional sensor installations. The system can also automatically notify relevant personnel in the event of a fire—complete with time, preset location, and a snapshot—allowing quick and effective response.

---

## Key Features

- Real-time fire and smoke detection in video feeds
- Supports standard CCTV footage
- Powered by YOLOv8 object detection
- Streamlit-based interactive web interface
- Upload and process video files
- Visual alerts and detection stats
- Adjustable confidence threshold
- Consecutive-frame verification to reduce false alarms
- Download processed, annotated videos
- Automatic alert system with:
  - Time of detection
  - Preset location of camera
  - Snapshot of fire/smoke detection frame

---

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

---

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/fire-smoke-detection.git
cd fire-smoke-detection
```

2. Install the required packages:
```
pip install -r requirements.txt
```

---

## Project Structure

```
fire-smoke-detection/
├── streamlit_app.py       # Streamlit web interface
├── train_model.py         # Training script
├── detect_fire_smoke.py   # Standalone detection
├── prepare_dataset.py     # Dataset prep utilities
├── data_utils.py          # Augmentation and extraction tools
├── notifier.py            # Automatic alerting system (NEW)
├── requirements.txt       
├── static/
│   └── results/           # Output videos
└── DATASET/               # Training dataset
```

---

## Dataset Format

```
DATASET/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**YOLO Label Format:**
```
class_id x_center y_center width height
```

- `class_id`: 0 = fire, 1 = smoke
- Coordinates and sizes normalized to [0, 1]

---

## Usage

### 1. Train the Model

```
python train_model.py --data_path DATASET --epochs 5 --batch 16
```

### 2. Run Detection on a Video

```
python detect_fire_smoke.py --video path/to/video.mp4 --model path/to/model.pt
```

Optional flags:
- `--output`: Output video path
- `--conf`: Confidence threshold
- `--no-display`: Disable live video preview

### 3. Run Web App

```
streamlit run streamlit_app.py
```

---

## Streamlit Interface Overview

- Left Panel: Upload, model selection, settings
- Right Panel: Video display, detection results
- Live Stats:
  - Detection counts
  - FPS
  - Progress and time remaining
- Post-processing:
  - Download results
  - Stop detection anytime

---

## Automatic Notification System

If a fire is detected:
- An alert message is automatically sent to emergency personnel.
- The message includes:
  - Time of detection
  - Camera location (predefined per feed)
  - Snapshot of the incident

---

## Model Training Results

Trained over 5 epochs:

| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 1     | 66.0% | 45.8%    |
| 2     | 74.1% | 53.1%    |
| 3     | 79.8% | 59.4%    |
| 4     | 82.2% | 63.2%    |
| 5     | 85.0% | 66.2%    |

Final Performance:
- Fire: 92.1% mAP50, 71.3% mAP50-95
- Smoke: 77.7% mAP50, 61.0% mAP50-95
- Precision: 87.4%, Recall: 83.2%

---

## Demo

### Input Video
<video src="Input.mp4" controls width="320" height="240"></video>

### Output Video
<video src="Output.mp4" controls width="320" height="240"></video>

---

## Understanding Detection Metrics

- **mAP50**: Detections with IoU ≥ 50% considered correct
- **mAP50-95**: Average mAP from IoU 50% to 95%
- **Precision**: Percentage of detections that were correct
- **Recall**: Percentage of actual instances that were detected

---

## Dataset Preparation Tips

Include:
- Fire images of varying environments, distances, and light
- Smoke in varying densities, colors, and sources
- False positive scenarios like fog, steam, or bright lights

---

## Data Utilities

```
# Manual annotation
python prepare_dataset.py annotate --output .

# Split into train/val/test
python prepare_dataset.py split --source_images images/ --source_labels labels/ --dest_dir DATASET --create_yaml

# Augment data
python data_utils.py augment --source your_dataset_dir --output augmented_data --factor 3

# Extract frames from video
python data_utils.py extract --video path/to/video.mp4 --output extracted_frames --interval 30
```

---

## Acknowledgements

This project was developed under the guidance and support of the faculty and peers at RV College of Engineering.

---

## Future Enhancements

- Integrate GPS or real-time camera location mapping
- Use real-world CCTV datasets
- Account for traffic, occlusion, and dynamic environments
- Integrate with city-wide alert systems or IoT networks

---

> Turning passive CCTV cameras into smart fire watchers.
