# Fire and Smoke Detection System

Real-time fire and smoke detection in videos using YOLOv8 and Streamlit.

## Overview

This project implements a real-time fire and smoke detection system using the YOLOv8 object detection model. It processes video files frame by frame to detect the presence of fire and smoke, offering visual alerts when either is detected in the video stream.

## Features

- Real-time detection of fire and smoke in video footage
- Interactive Streamlit web interface
- Upload and process video files
- Visual alerts for fire and smoke detection
- Performance metrics (FPS, detection counts)
- Adjustable confidence threshold for detection sensitivity
- Consecutive frame verification to reduce false alarms
- Download processed videos with detection annotations

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time performance, but works on CPU)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/fire-smoke-detection.git
cd fire-smoke-detection
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Project Structure

```
fire-smoke-detection/
├── streamlit_app.py       # Streamlit web application
├── train_model.py         # Training script for YOLOv8
├── detect_fire_smoke.py   # Standalone detection script
├── prepare_dataset.py     # Dataset preparation utilities
├── data_utils.py          # Data augmentation utilities
├── requirements.txt       # Project dependencies
├── static/                # Static files for web app
│   └── results/           # Folder for processed videos
└── DATASET/               # Training dataset folder
```

## Dataset Structure

The project is configured to work with your existing dataset structure:

```
DATASET/
├── train/
│   ├── images/ (training images)
│   └── labels/ (training labels)
├── valid/
│   ├── images/ (validation images)
│   └── labels/ (validation labels)
├── test/
│   ├── images/ (test images)
│   └── labels/ (test labels)
└── data.yaml (dataset configuration)
```

### Label Format

Labels should be provided in YOLO format (one .txt file per image) with the following format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: 0 for fire, 1 for smoke
- `x_center`, `y_center`: normalized center coordinates (0-1)
- `width`, `height`: normalized width and height (0-1)

## Usage

### 1. Train the Model

To train the YOLOv8 model with your custom dataset:

```
python train_model.py --data_path DATASET --epochs 5 --batch 16
```

Parameters:
- `--data_path`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 5)
- `--batch`: Batch size (default: 16)
- `--img_size`: Image size for training (default: 640)
- `--weights`: Initial weights for training (default: 'yolov8m.pt')

### 2. Run Detection on a Single Video

To process a single video file without the web interface:

```
python detect_fire_smoke.py --video path/to/video.mp4 --model path/to/model.pt
```

Parameters:
- `--video`: Path to the input video file
- `--model`: Path to the YOLOv8 model (default: 'runs/detect/fire_smoke_detection2/weights/best.pt')
- `--output`: Path to save the output video (optional)
- `--conf`: Confidence threshold (default: 0.5)
- `--no-display`: Run without displaying the video

### 3. Run the Streamlit Web Interface

```
streamlit run streamlit_app.py
```

The Streamlit interface will automatically open in your default web browser.

## Streamlit Interface Usage

1. The interface is divided into two main sections:
   - Left panel: Upload controls and detection statistics
   - Right panel: Video display and results

2. Upload a video using the file uploader

3. Configure detection settings in the sidebar:
   - Select the model to use
   - Adjust confidence threshold
   - Set display options
   - Configure alert settings

4. Click "Start Detection" to begin processing

5. Monitor real-time statistics:
   - Fire and smoke detection counts
   - Processing FPS
   - Progress bar
   - Elapsed and remaining time

6. After processing completes:
   - View the final statistics
   - Download the processed video with annotations

7. Click "Stop Detection" at any time to halt processing

## Model Training Results

The YOLOv8 model was trained for 5 epochs with the following results:

- **Epoch 1**: mAP50 66.0%, mAP50-95 45.8%
- **Epoch 2**: mAP50 74.1%, mAP50-95 53.1%
- **Epoch 3**: mAP50 79.8%, mAP50-95 59.4%
- **Epoch 4**: mAP50 82.2%, mAP50-95 63.2%
- **Epoch 5**: mAP50 85.0%, mAP50-95 66.2%

Final model performance:
- Fire class: 92.1% mAP50, 71.3% mAP50-95
- Smoke class: 77.7% mAP50, 61.0% mAP50-95
- Overall precision: 87.4%, recall: 83.2%

### Input
- `Sample Video.mp4`
  <video src="Sample Video.mp4" controls width="320" height="240"></video>

### Output
- `output_20250614_131203.mp4`
  <video src="output_20250614_131203.mp4" controls width="320" height="240"></video>

## Understanding Detection Metrics

- **mAP50**: Mean Average Precision at 50% IoU threshold - accuracy when considering detections with 50% overlap with ground truth as correct
- **mAP50-95**: Average mAP across multiple IoU thresholds from 50% to 95% - a stricter metric requiring more precise bounding boxes
- **Precision**: When the model predicts fire/smoke, how often it's correct (low false positives)
- **Recall**: How many actual fire/smoke instances the model finds (low false negatives)

## Tips for Dataset Creation

For best detection results, include:

1. **Diverse Fire Images**:
   - Different types of fires (indoor, outdoor, structural)
   - Various sizes and distances
   - Different lighting conditions

2. **Diverse Smoke Images**:
   - Various smoke types (white, gray, black)
   - Different densities and distances
   - Smoke with and without visible fire source

3. **Negative Examples**:
   - Red/orange objects that aren't fire
   - Fog/steam/clouds that aren't smoke
   - Various lighting conditions that might be confused with fire/smoke

## Data Preparation Utilities

This project includes utilities for dataset preparation:

1. **Annotation Tool**: Create a manual annotation tool for labeling images
   ```
   python prepare_dataset.py annotate --output .
   ```

2. **Dataset Splitting**: Split a collection of images and labels into train/val/test sets
   ```
   python prepare_dataset.py split --source_images images/ --source_labels labels/ --dest_dir DATASET --create_yaml
   ```

3. **Data Augmentation**: Augment your existing dataset to increase diversity
   ```
   python data_utils.py augment --source your_dataset_dir --output augmented_data --factor 3
   ```

4. **Frame Extraction**: Extract frames from videos for annotation
   ```
   python data_utils.py extract --video path/to/video.mp4 --output extracted_frames --interval 30
   ```
