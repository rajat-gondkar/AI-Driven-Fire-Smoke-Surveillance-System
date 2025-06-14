#!/usr/bin/env python3
"""
Dataset Preparation Tool for Fire and Smoke Detection

This script helps prepare and organize images and labels for training 
a YOLOv8 fire and smoke detection model.
"""

import os
import shutil
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml

def create_dir_structure(root_dir):
    """Create the required directory structure for YOLOv8 training"""
    # Create main directories
    for subdir in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(root_dir, subdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, subdir, 'labels'), exist_ok=True)
    
    print(f"Created directory structure at: {root_dir}")
    return {
        'images': {
            'train': os.path.join(root_dir, 'train', 'images'),
            'val': os.path.join(root_dir, 'valid', 'images'),
            'test': os.path.join(root_dir, 'test', 'images')
        },
        'labels': {
            'train': os.path.join(root_dir, 'train', 'labels'),
            'val': os.path.join(root_dir, 'valid', 'labels'),
            'test': os.path.join(root_dir, 'test', 'labels')
        }
    }

def split_dataset(source_images, source_labels, dest_dirs, split_ratio=(0.7, 0.2, 0.1), copy=True):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        source_images: Directory containing source images
        source_labels: Directory containing source labels (same names as images but .txt extension)
        dest_dirs: Dictionary with destination directories
        split_ratio: Tuple of (train, val, test) ratios, should sum to 1
        copy: Whether to copy files (True) or move them (False)
    """
    # Validate split ratio
    if sum(split_ratio) != 1.0:
        print("Warning: Split ratios don't sum to 1.0, normalizing...")
        split_ratio = tuple(r / sum(split_ratio) for r in split_ratio)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(source_images).glob(ext)))
    
    # Shuffle the files for randomness
    random.shuffle(image_files)
    
    # Calculate split counts
    n_files = len(image_files)
    n_train = int(n_files * split_ratio[0])
    n_val = int(n_files * split_ratio[1])
    # n_test will be the remainder
    
    print(f"Found {n_files} images, splitting into:")
    print(f"  - Train: {n_train}")
    print(f"  - Val: {n_val}")
    print(f"  - Test: {n_files - n_train - n_val}")
    
    # Split the image files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Function to copy or move files
    def transfer_files(files, dest_img_dir, dest_lbl_dir):
        for img_path in tqdm(files):
            # Get corresponding label file path
            label_path = Path(source_labels) / f"{img_path.stem}.txt"
            
            # Define destination paths
            dest_img_path = os.path.join(dest_img_dir, img_path.name)
            dest_lbl_path = os.path.join(dest_lbl_dir, f"{img_path.stem}.txt")
            
            # Copy or move image file
            if copy:
                shutil.copy2(img_path, dest_img_path)
            else:
                shutil.move(img_path, dest_img_path)
            
            # Copy or move label file if it exists
            if label_path.exists():
                if copy:
                    shutil.copy2(label_path, dest_lbl_path)
                else:
                    shutil.move(label_path, dest_lbl_path)
            else:
                print(f"Warning: Label file not found for {img_path.name}")
    
    # Process files for each split
    print("Processing train set...")
    transfer_files(train_files, dest_dirs['images']['train'], dest_dirs['labels']['train'])
    
    print("Processing validation set...")
    transfer_files(val_files, dest_dirs['images']['val'], dest_dirs['labels']['val'])
    
    print("Processing test set...")
    transfer_files(test_files, dest_dirs['images']['test'], dest_dirs['labels']['test'])
    
    print("Dataset splitting complete!")

def convert_bbox_format(image_path, input_bbox, format_type):
    """
    Convert bounding box formats
    
    Args:
        image_path: Path to the image file
        input_bbox: Input bounding box [x, y, width, height] or [x1, y1, x2, y2]
        format_type: Either 'voc_to_yolo' or 'yolo_to_voc'
    
    Returns:
        Converted bounding box
    """
    # Read image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    if format_type == 'voc_to_yolo':
        # VOC format: [xmin, ymin, xmax, ymax]
        # YOLO format: [x_center, y_center, width, height] (normalized)
        x_min, y_min, x_max, y_max = input_bbox
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
        
    elif format_type == 'yolo_to_voc':
        # YOLO format: [x_center, y_center, width, height] (normalized)
        # VOC format: [xmin, ymin, xmax, ymax]
        x_center, y_center, width, height = input_bbox
        
        # Convert to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Convert to VOC format
        x_min = int(x_center - (width / 2))
        y_min = int(y_center - (height / 2))
        x_max = int(x_center + (width / 2))
        y_max = int(y_center + (height / 2))
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        return [x_min, y_min, x_max, y_max]
    else:
        raise ValueError("format_type must be either 'voc_to_yolo' or 'yolo_to_voc'")

def create_manual_annotation_tool(output_dir):
    """
    Create a simple manual annotation script to help create labels
    
    Args:
        output_dir: Directory where the annotation script will be saved
    """
    annotation_tool_code = """
import cv2
import os
import argparse
from pathlib import Path

class ManualAnnotator:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.current_class = 0  # 0 for fire, 1 for smoke
        self.current_image_idx = 0
        self.drawing = False
        self.images = []
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.window_name = "Fire and Smoke Annotator"
        self.load_images()
        
    def load_images(self):
        """Load all image paths from the directory"""
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.images.extend(list(Path(self.images_dir).glob(ext)))
        self.images.sort()
        
        if not self.images:
            print(f"No images found in {self.images_dir}")
            exit(1)
            
        print(f"Loaded {len(self.images)} images")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Create a copy of the original image to draw on
            img_copy = self.current_image.copy()
            cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, img_copy)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Draw the final rectangle
            cv2.rectangle(self.current_image, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_image)
            
            # Save the annotation in YOLO format
            self.save_annotation()
    
    def save_annotation(self):
        """Save the current annotation in YOLO format"""
        img_height, img_width = self.current_image_original.shape[:2]
        
        # Ensure points are ordered correctly
        x_min = min(self.start_point[0], self.end_point[0])
        y_min = min(self.start_point[1], self.end_point[1])
        x_max = max(self.start_point[0], self.end_point[0])
        y_max = max(self.start_point[1], self.end_point[1])
        
        # Convert to YOLO format (normalized)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Create label file path
        img_name = self.images[self.current_image_idx].stem
        label_path = os.path.join(self.labels_dir, f"{img_name}.txt")
        
        # Append to label file (can have multiple objects)
        with open(label_path, 'a') as f:
            f.write(f"{self.current_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
            
        print(f"Annotation saved: class={self.current_class}, box=[{x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}]")
        
    def run(self):
        """Run the annotation tool"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            # Load current image
            if 0 <= self.current_image_idx < len(self.images):
                img_path = str(self.images[self.current_image_idx])
                self.current_image_original = cv2.imread(img_path)
                self.current_image = self.current_image_original.copy()
                
                # Display class info
                class_text = f"Class: {'Fire' if self.current_class == 0 else 'Smoke'} (press 'c' to change)"
                cv2.putText(self.current_image, class_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display navigation info
                nav_text = f"Image {self.current_image_idx + 1}/{len(self.images)} (use left/right arrows to navigate)"
                cv2.putText(self.current_image, nav_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display instructions
                cv2.putText(self.current_image, "Draw boxes with mouse. 'r' to reset, 'd' to delete label, 'q' to quit", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Check if label file exists and show existing annotations
                img_name = self.images[self.current_image_idx].stem
                label_path = os.path.join(self.labels_dir, f"{img_name}.txt")
                if os.path.exists(label_path):
                    self.display_existing_annotations(label_path)
                
                cv2.imshow(self.window_name, self.current_image)
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
                
            elif key == ord('c'):  # Change class
                self.current_class = 1 - self.current_class  # Toggle between 0 and 1
                print(f"Changed to {'Fire' if self.current_class == 0 else 'Smoke'} class")
                # Refresh the display
                self.current_image = self.current_image_original.copy()
                cv2.imshow(self.window_name, self.current_image)
                
            elif key == 81:  # Left arrow - previous image
                self.current_image_idx = max(0, self.current_image_idx - 1)
                
            elif key == 83:  # Right arrow - next image
                self.current_image_idx = min(len(self.images) - 1, self.current_image_idx + 1)
                
            elif key == ord('r'):  # Reset current image
                if os.path.exists(label_path):
                    with open(label_path, 'w') as f:
                        pass  # Clear file contents
                    print(f"Reset annotations for {img_name}")
                    self.current_image = self.current_image_original.copy()
                    cv2.imshow(self.window_name, self.current_image)
                    
            elif key == ord('d'):  # Delete label file
                img_name = self.images[self.current_image_idx].stem
                label_path = os.path.join(self.labels_dir, f"{img_name}.txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Deleted label file for {img_name}")
                
        cv2.destroyAllWindows()
    
    def display_existing_annotations(self, label_path):
        """Display existing annotations on the current image"""
        img_height, img_width = self.current_image.shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class x_center y_center width height
                    cls, x_center, y_center, width, height = parts
                    cls = int(cls)
                    x_center = float(x_center) * img_width
                    y_center = float(y_center) * img_height
                    width = float(width) * img_width
                    height = float(height) * img_height
                    
                    # Calculate rectangle coordinates
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    # Draw rectangle with color based on class
                    color = (0, 0, 255) if cls == 0 else (128, 128, 128)  # Red for fire, gray for smoke
                    cv2.rectangle(self.current_image, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Add label
                    label = "Fire" if cls == 0 else "Smoke"
                    cv2.putText(self.current_image, label, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    parser = argparse.ArgumentParser(description="Manual annotation tool for fire and smoke detection")
    parser.add_argument("--images", type=str, required=True, 
                        help="Directory containing images to annotate")
    parser.add_argument("--labels", type=str, required=True,
                        help="Directory where label files will be saved")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.labels, exist_ok=True)
    
    # Create and run annotator
    annotator = ManualAnnotator(args.images, args.labels)
    annotator.run()

if __name__ == "__main__":
    main()
"""
    
    tool_path = os.path.join(output_dir, 'annotate.py')
    with open(tool_path, 'w') as f:
        f.write(annotation_tool_code)
    
    print(f"Created annotation tool at: {tool_path}")
    print("Usage: python annotate.py --images <images_dir> --labels <labels_dir>")

def create_data_yaml(root_dir):
    """Create a data.yaml file for YOLOv8 training"""
    yaml_content = {
        'train': os.path.join(root_dir, 'train', 'images'),
        'val': os.path.join(root_dir, 'valid', 'images'),
        'test': os.path.join(root_dir, 'test', 'images'),
        'nc': 2,
        'names': ['Fire', 'Smoke']
    }
    
    yaml_path = os.path.join(root_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {yaml_path}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation for fire and smoke detection")
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create directory structure command
    create_parser = subparsers.add_parser("create", help="Create dataset directory structure")
    create_parser.add_argument("--dir", type=str, default="DATASET",
                             help="Root directory for the dataset")
    create_parser.add_argument("--create_yaml", action="store_true",
                             help="Create data.yaml file")
    
    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split dataset into train, val, test")
    split_parser.add_argument("--source_images", type=str, required=True,
                             help="Directory containing source images")
    split_parser.add_argument("--source_labels", type=str, required=True,
                             help="Directory containing source labels")
    split_parser.add_argument("--dest_dir", type=str, default="DATASET",
                             help="Destination root directory")
    split_parser.add_argument("--train", type=float, default=0.7,
                             help="Proportion for training set")
    split_parser.add_argument("--val", type=float, default=0.2,
                             help="Proportion for validation set")
    split_parser.add_argument("--test", type=float, default=0.1,
                             help="Proportion for test set")
    split_parser.add_argument("--move", action="store_true",
                             help="Move files instead of copying")
    split_parser.add_argument("--create_yaml", action="store_true",
                             help="Create data.yaml file after splitting")
    
    # Generate annotation tool command
    annotate_parser = subparsers.add_parser("annotate", help="Generate annotation tool")
    annotate_parser.add_argument("--output", type=str, default=".",
                                help="Directory where to save the annotation tool")
    
    args = parser.parse_args()
    
    # Execute the selected command
    if args.command == "create":
        create_dir_structure(args.dir)
        if args.create_yaml:
            create_data_yaml(args.dir)
    
    elif args.command == "split":
        # Create the destination directory structure
        dest_dirs = create_dir_structure(args.dest_dir)
        
        # Split the dataset
        split_dataset(args.source_images, args.source_labels, dest_dirs, 
                      split_ratio=(args.train, args.val, args.test),
                      copy=not args.move)
        
        # Create data.yaml if requested
        if args.create_yaml:
            create_data_yaml(args.dest_dir)
    
    elif args.command == "annotate":
        create_manual_annotation_tool(args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 