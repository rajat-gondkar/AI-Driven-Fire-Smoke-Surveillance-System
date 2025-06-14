import cv2
import numpy as np
import argparse
import time
import os
from ultralytics import YOLO
from pathlib import Path

def detect_fire_smoke(video_path, model_path, conf_threshold=0.5, output_path=None, show_video=True):
    """
    Detect fire and smoke in a video using YOLOv8 model
    
    Args:
        video_path: Path to the input video
        model_path: Path to the YOLOv8 model
        conf_threshold: Confidence threshold for detections
        output_path: Path to save the output video (optional)
        show_video: Whether to display the video during processing
        
    Returns:
        Dictionary with detection statistics
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Open the video
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer if output path is specified
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize counters
    frame_count = 0
    fire_detected = 0
    smoke_detected = 0
    processing_times = []
    
    # Process the video
    print("Starting detection...")
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start time for FPS calculation
        start_time = time.time()
        
        # Perform detection
        results = model(frame, conf=conf_threshold)
        
        # Process results
        result_frame = results[0].plot()
        
        # Count detections
        for detection in results[0].boxes.data.tolist():
            class_id = int(detection[5])
            confidence = detection[4]
            
            if class_id == 0:  # Fire class
                fire_detected += 1
            elif class_id == 1:  # Smoke class
                smoke_detected += 1
        
        # Calculate processing time
        process_time = time.time() - start_time
        processing_times.append(process_time)
        current_fps = 1 / process_time if process_time > 0 else 0
        
        # Display FPS and progress on the frame
        cv2.putText(result_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        progress = frame_count / total_frames * 100
        cv2.putText(result_frame, f"Progress: {progress:.1f}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output video if specified
        if output_path:
            out.write(result_frame)
        
        # Display the frame
        if show_video:
            cv2.imshow("Fire and Smoke Detection", result_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update frame count
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    avg_fps = 1 / (sum(processing_times) / len(processing_times)) if processing_times else 0
    stats = {
        "total_frames": frame_count,
        "fire_detections": fire_detected,
        "smoke_detections": smoke_detected,
        "avg_fps": avg_fps,
        "processing_time": sum(processing_times)
    }
    
    # Print summary
    print("\n--- Detection Summary ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Fire detections: {fire_detected}")
    print(f"Smoke detections: {smoke_detected}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total processing time: {sum(processing_times):.2f} seconds")
    
    if output_path:
        print(f"Output video saved to: {output_path}")
    
    return stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect fire and smoke in videos using YOLOv8")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="runs/detect/fire_smoke_detection2/weights/best.pt", 
                        help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    parser.add_argument("--no-display", action="store_true", help="Do not display video during processing")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist")
        return
    
    # Run detection
    detect_fire_smoke(
        video_path=args.video,
        model_path=args.model,
        conf_threshold=args.conf,
        output_path=args.output,
        show_video=not args.no_display
    )

if __name__ == "__main__":
    main() 