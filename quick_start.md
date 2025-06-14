# Fire and Smoke Detection System - Quick Start Guide

This guide will help you quickly set up and run the fire and smoke detection system.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Note that the system is configured to work with your existing dataset structure:
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

## Dataset Preparation

### Option 1: Use existing images with manual annotation

1. Place your fire and smoke images in a directory
2. Run the annotation tool to create labels:
   ```
   python prepare_dataset.py annotate --output .
   python annotate.py --images your_images_dir --labels your_labels_dir
   ```
3. Follow the on-screen instructions to draw bounding boxes around fire (class 0) and smoke (class 1)

### Option 2: Extract frames from videos

1. Extract frames from fire and smoke videos:
   ```
   python data_utils.py extract --video path/to/video.mp4 --output extracted_frames --interval 30
   ```
2. Annotate the extracted frames as described in Option 1

### Option 3: Augment existing labeled dataset

1. If you already have some labeled images, augment them to increase the dataset size:
   ```
   python data_utils.py augment --source your_dataset_dir --output augmented_data --factor 3
   ```

### Organize the dataset

1. Split your annotated data into training, validation, and testing sets:
   ```
   python prepare_dataset.py split --source_images your_images_dir --source_labels your_labels_dir --dest_dir DATASET --create_yaml
   ```

## Training

Train the YOLOv8 model on your prepared dataset:

```
python train_model.py --data_path DATASET --epochs 5 --batch 16
```

You can adjust the number of epochs and batch size based on your hardware capabilities. The default values are:
- epochs: 5 (increase for better accuracy)
- batch: 16 (decrease if you encounter memory issues)
- img_size: 640 (standard YOLOv8 input size)

## Running Detection

### Option 1: Command line detection

To run detection on a video file from the command line:

```
python detect_fire_smoke.py --video path/to/video.mp4 --model runs/detect/fire_smoke_detection/weights/best.pt
```

Options:
- `--output path/to/save/output.mp4`: Save the processed video
- `--conf 0.5`: Adjust confidence threshold (0.1-0.9)
- `--no-display`: Process without displaying the video

### Option 2: Web interface

Start the web application:

```
python app.py
```

Then open your browser and navigate to: http://localhost:5000

Through the web interface, you can:
1. Upload videos for processing
2. Adjust the confidence threshold using a slider
3. View real-time detection results
4. Get alerts when fire or smoke is detected

## Tips for Best Results

1. **Diverse Dataset**: Include images of fire and smoke in various environments, lighting conditions, and distances
2. **Balance Classes**: Try to maintain a balance between fire and smoke examples
3. **Negative Examples**: Include images that might cause false positives (red/orange objects, fog, steam)
4. **Confidence Threshold**: Adjust the confidence threshold based on your needs:
   - Lower threshold (0.3-0.4): More detections, potential false positives
   - Higher threshold (0.6-0.8): Fewer detections, but more reliable

## Troubleshooting

1. **CUDA out of memory**: Try reducing batch size or image size during training
2. **Low FPS during detection**: Consider using a smaller model (YOLOv8n or YOLOv8s) or reducing the input size
3. **False positives**: Increase the confidence threshold and add more training data with negative examples
4. **False negatives**: Lower the confidence threshold and add more diverse training examples 