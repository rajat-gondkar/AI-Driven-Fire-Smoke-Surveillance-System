import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_folder, quality=95, resize_factor=1.0):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting frames to: {output_folder}")
    print(f"Compression quality: {quality}%")
    print(f"Resize factor: {resize_factor}")
    
    # Read and save frames
    frame_number = 0
    while True:
        # Read next frame
        success, frame = video.read()
        if not success:
            break
        
        # Resize if needed
        if resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * resize_factor)
            new_width = int(width * resize_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save frame as image with compression
        filename = os.path.join(output_folder, f"frame_{frame_number:05d}.jpg")
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(filename, frame, encode_params)
        
        # Update progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames ({(frame_number/frame_count*100):.1f}%)")
            
        frame_number += 1
    
    # Release video
    video.release()
    print(f"Done! Extracted {frame_number} frames to {output_folder}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", default="frames", help="Output folder for frames (default: 'frames')")
    parser.add_argument("--quality", "-q", type=int, default=95, 
                        help="JPEG compression quality (0-100, default: 95)")
    parser.add_argument("--resize", "-r", type=float, default=1.0,
                        help="Resize factor (e.g., 0.5 for half size, default: 1.0)")
    args = parser.parse_args()
    
    # Extract frames
    extract_frames(args.video_path, args.output, args.quality, args.resize) 