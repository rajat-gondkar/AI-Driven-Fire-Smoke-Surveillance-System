#!/usr/bin/env python3
"""
Data Augmentation Utilities for Fire and Smoke Detection

This script provides utilities for augmenting fire and smoke images
to improve model training and generalization.
"""

import os
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
import random
from pathlib import Path
import shutil

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust image brightness and contrast
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted image
    """
    # Convert to float for calculations
    img_float = image.astype(float)
    
    # Apply brightness
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        img_float = alpha_b * img_float + gamma_b
    
    # Apply contrast
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        img_float = alpha_c * img_float + gamma_c
    
    # Clip values to valid range
    img_float = np.clip(img_float, 0, 255)
    
    return img_float.astype(np.uint8)

def add_noise(image, noise_type="gaussian", amount=10):
    """
    Add noise to image
    
    Args:
        image: Input image
        noise_type: Type of noise to add ('gaussian', 'salt_pepper')
        amount: Noise amount
        
    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = amount
        noise = np.random.normal(mean, sigma, (row, col, ch))
        noisy_img = image + noise.astype(np.uint8)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = amount / 100  # Convert to percentage
        
        noisy_img = np.copy(image)
        
        # Salt (white) mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = 255
        
        # Pepper (black) mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[coords[0], coords[1], :] = 0
        
        return noisy_img
    
    else:
        return image

def rotate_image_and_labels(image, labels, angle):
    """
    Rotate image and update bounding box labels accordingly
    
    Args:
        image: Input image
        labels: List of labels in YOLO format [class_id, x_center, y_center, width, height]
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image and updated labels
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate new image size to avoid cropping
    diagonal = np.sqrt(height**2 + width**2)
    new_height = int(diagonal)
    new_width = int(diagonal)
    
    # Compute rotation matrix
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust rotation matrix to avoid cropping
    rot_mat[0, 2] += (new_width - width) / 2
    rot_mat[1, 2] += (new_height - height) / 2
    
    # Perform rotation
    rotated_img = cv2.warpAffine(image, rot_mat, (new_width, new_height))
    
    # Update labels
    updated_labels = []
    for label in labels:
        class_id = label[0]
        x_center = label[1]
        y_center = label[2]
        box_width = label[3]
        box_height = label[4]
        
        # Convert normalized coordinates to absolute
        abs_x = x_center * width
        abs_y = y_center * height
        
        # Get box corners (absolute coordinates)
        x1 = abs_x - box_width * width / 2
        y1 = abs_y - box_height * height / 2
        x2 = abs_x + box_width * width / 2
        y2 = abs_y + box_height * height / 2
        
        # Create a list of the box's corner points
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        # Rotate corners
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        rotated_corners = corners_ones.dot(rot_mat.T)
        
        # Get new bounding box
        min_x = np.min(rotated_corners[:, 0])
        min_y = np.min(rotated_corners[:, 1])
        max_x = np.max(rotated_corners[:, 0])
        max_y = np.max(rotated_corners[:, 1])
        
        # Convert back to normalized YOLO format
        new_x_center = (min_x + max_x) / 2 / new_width
        new_y_center = (min_y + max_y) / 2 / new_height
        new_width_norm = (max_x - min_x) / new_width
        new_height_norm = (max_y - min_y) / new_height
        
        # Ensure values are in valid range
        new_x_center = max(0, min(1, new_x_center))
        new_y_center = max(0, min(1, new_y_center))
        new_width_norm = max(0, min(1, new_width_norm))
        new_height_norm = max(0, min(1, new_height_norm))
        
        # Add updated label
        updated_labels.append([class_id, new_x_center, new_y_center, new_width_norm, new_height_norm])
    
    return rotated_img, updated_labels

def flip_image_and_labels(image, labels, flip_code):
    """
    Flip image horizontally or vertically and update bounding box labels
    
    Args:
        image: Input image
        labels: List of labels in YOLO format [class_id, x_center, y_center, width, height]
        flip_code: Flip direction (0 for vertical, 1 for horizontal)
        
    Returns:
        Flipped image and updated labels
    """
    # Flip the image
    flipped_img = cv2.flip(image, flip_code)
    
    # Update labels
    updated_labels = []
    for label in labels:
        class_id = label[0]
        x_center = label[1]
        y_center = label[2]
        box_width = label[3]
        box_height = label[4]
        
        if flip_code == 1:  # Horizontal flip
            new_x_center = 1.0 - x_center
            new_y_center = y_center
        else:  # Vertical flip
            new_x_center = x_center
            new_y_center = 1.0 - y_center
        
        updated_labels.append([class_id, new_x_center, new_y_center, box_width, box_height])
    
    return flipped_img, updated_labels

def augment_dataset(source_dir, output_dir, augmentation_factor=3):
    """
    Augment dataset with various transformations
    
    Args:
        source_dir: Directory containing source images and labels
        output_dir: Directory to save augmented images and labels
        augmentation_factor: Number of augmented versions to create per image
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(source_dir, 'images', ext)))
    
    print(f"Found {len(image_files)} images to augment")
    
    # Process each image
    for img_path in tqdm(image_files):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Check if label file exists
        label_path = os.path.join(source_dir, 'labels', f"{base_name}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {base_name}, skipping")
            continue
        
        # Load labels
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0]), float(parts[1]), float(parts[2]), 
                                  float(parts[3]), float(parts[4])])
        
        # Copy original files
        shutil.copy2(img_path, os.path.join(images_dir, f"{base_name}_orig.jpg"))
        shutil.copy2(label_path, os.path.join(labels_dir, f"{base_name}_orig.txt"))
        
        # Generate augmented versions
        for i in range(augmentation_factor):
            # Random augmentation parameters
            brightness = random.randint(-50, 50)
            contrast = random.randint(-50, 50)
            noise_amount = random.randint(5, 20)
            flip_code = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
            rotation_angle = random.choice([0, 90, 180, 270])  # Limit to 90-degree rotations
            
            # Apply transformations
            img_aug = img.copy()
            labels_aug = labels.copy()
            
            # Apply brightness/contrast adjustment
            img_aug = adjust_brightness_contrast(img_aug, brightness, contrast)
            
            # Apply noise if chance is met
            if random.random() > 0.5:
                noise_type = random.choice(["gaussian", "salt_pepper"])
                img_aug = add_noise(img_aug, noise_type, noise_amount)
            
            # Apply flip if chance is met
            if random.random() > 0.5 and flip_code != -1:
                img_aug, labels_aug = flip_image_and_labels(img_aug, labels_aug, flip_code)
            
            # Apply rotation if chance is met
            if random.random() > 0.5 and rotation_angle != 0:
                img_aug, labels_aug = rotate_image_and_labels(img_aug, labels_aug, rotation_angle)
            
            # Save augmented image
            aug_name = f"{base_name}_aug{i}"
            cv2.imwrite(os.path.join(images_dir, f"{aug_name}.jpg"), img_aug)
            
            # Save augmented labels
            with open(os.path.join(labels_dir, f"{aug_name}.txt"), 'w') as f:
                for label in labels_aug:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
    
    print(f"Augmentation complete. Created {len(image_files) * augmentation_factor} augmented images.")

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting frames from {video_path} (every {frame_interval} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Data augmentation utilities for fire and smoke detection")
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Augmentation command
    augment_parser = subparsers.add_parser("augment", help="Augment dataset")
    augment_parser.add_argument("--source", type=str, required=True,
                              help="Source directory with images/ and labels/ subdirectories")
    augment_parser.add_argument("--output", type=str, required=True,
                              help="Output directory for augmented data")
    augment_parser.add_argument("--factor", type=int, default=3,
                              help="Augmentation factor (num of augmented versions per image)")
    
    # Frame extraction command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from video")
    extract_parser.add_argument("--video", type=str, required=True,
                               help="Path to input video file")
    extract_parser.add_argument("--output", type=str, required=True,
                               help="Directory to save extracted frames")
    extract_parser.add_argument("--interval", type=int, default=30,
                               help="Extract every nth frame")
    
    args = parser.parse_args()
    
    # Execute the selected command
    if args.command == "augment":
        augment_dataset(args.source, args.output, args.factor)
    
    elif args.command == "extract":
        extract_frames(args.video, args.output, args.interval)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 