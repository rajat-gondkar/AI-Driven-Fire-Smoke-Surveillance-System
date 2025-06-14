import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image
import datetime

# Set page configuration
st.set_page_config(
    page_title="Fire and Smoke Detection",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4A4A4A;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .alert-fire {
        background-color: rgba(255, 87, 51, 0.1);
        border-left: 5px solid #FF5733;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alert-smoke {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 5px solid #808080;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF5733;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #eee;
        color: #6c757d;
        font-size: 0.9rem;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing detection results and processing status
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'fire_detections' not in st.session_state:
    st.session_state.fire_detections = 0
if 'smoke_detections' not in st.session_state:
    st.session_state.smoke_detections = 0
if 'current_fps' not in st.session_state:
    st.session_state.current_fps = 0
if 'fire_detected' not in st.session_state:
    st.session_state.fire_detected = False
if 'smoke_detected' not in st.session_state:
    st.session_state.smoke_detected = False
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0

# Header
st.markdown('<h1 class="main-header">🔥 Fire and Smoke Detection System</h1>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Function to resize frame for faster processing
def resize_frame(frame, scale_factor):
    if scale_factor == 1.0:
        return frame
    
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    return cv2.resize(frame, (width, height))

# Function to format time
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<h2 class="sub-header">Upload Video</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    # Model selection
    model_path = st.selectbox(
        "Select Model",
        [
            "runs/detect/fire_smoke_detection2/weights/best.pt",
            "runs/detect/fire_smoke_detection2/weights/last.pt",
        ],
        index=0
    )
    
    # Confidence threshold
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Performance options
    st.subheader("Performance Options")
    frame_skip = st.slider("Frame Skip (higher = faster)", 0, 5, 2, 1)
    resize_factor = st.slider("Resolution Scale (lower = faster)", 0.25, 1.0, 0.5, 0.25)
    
    # Display options
    st.subheader("Display Options")
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    display_fps = st.checkbox("Display FPS", value=True)
    
    # Process buttons
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")
    
    # Display alert status
    if st.session_state.fire_detected:
        st.markdown('<div class="alert-fire"><strong>⚠️ FIRE DETECTED!</strong> Take immediate action!</div>', unsafe_allow_html=True)
    
    if st.session_state.smoke_detected:
        st.markdown('<div class="alert-smoke"><strong>⚠️ SMOKE DETECTED!</strong> Investigate immediately!</div>', unsafe_allow_html=True)
    
    # Display metrics
    if st.session_state.processed_frames > 0:
        st.markdown("### Detection Statistics")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.fire_detections}</div>
                <div class="metric-label">Fire Detections</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.smoke_detections}</div>
                <div class="metric-label">Smoke Detections</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.current_fps:.1f}</div>
                <div class="metric-label">FPS</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display progress
        progress_percentage = st.session_state.processed_frames / st.session_state.total_frames if st.session_state.total_frames > 0 else 0
        st.progress(min(progress_percentage, 1.0))

with col2:
    st.markdown('<h2 class="sub-header">Detection View</h2>', unsafe_allow_html=True)
    
    # Display area for video
    video_display = st.empty()

# Process video function
def process_video(video_path, model, conf_threshold):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.total_frames = total_frames
    
    # Reset detection status
    st.session_state.fire_detected = False
    st.session_state.smoke_detected = False
    st.session_state.fire_detections = 0
    st.session_state.smoke_detections = 0
    st.session_state.processed_frames = 0
    
    # Process the video
    frame_idx = 0
    skip_counter = 0
    
    while cap.isOpened() and st.session_state.processing:
        # Start time for FPS calculation
        start_time = time.time()
        
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for performance if needed
        if frame_skip > 0:
            skip_counter = (skip_counter + 1) % (frame_skip + 1)
            if skip_counter != 0:
                frame_idx += 1
                continue
        
        # Store original size
        original_size = frame.shape
        
        # Resize frame for faster processing
        resized_frame = resize_frame(frame, resize_factor)
        
        # Perform detection
        results = model(resized_frame, conf=conf_threshold)
        
        # Process results
        result_frame = results[0].plot(labels=show_labels, conf=show_conf)
        
        # Count detections in this frame
        frame_fire_count = 0
        frame_smoke_count = 0
        
        for detection in results[0].boxes.data.tolist():
            class_id = int(detection[5])
            confidence = detection[4]
            
            if class_id == 0:  # Fire class
                frame_fire_count += 1
                st.session_state.fire_detected = True
            elif class_id == 1:  # Smoke class
                frame_smoke_count += 1
                st.session_state.smoke_detected = True
        
        # Update statistics
        st.session_state.fire_detections += frame_fire_count
        st.session_state.smoke_detections += frame_smoke_count
        st.session_state.processed_frames += 1
        
        # Calculate FPS
        process_time = time.time() - start_time
        current_fps = 1 / process_time if process_time > 0 else 0
        st.session_state.current_fps = current_fps
        
        # Add alert indicators to the frame
        if st.session_state.fire_detected:
            cv2.putText(result_frame, "FIRE ALERT!", (result_frame.shape[1] - 250, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if st.session_state.smoke_detected:
            cv2.putText(result_frame, "SMOKE ALERT!", (result_frame.shape[1] - 250, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        
        # Display FPS on the frame
        if display_fps:
            cv2.putText(result_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # If frame was resized, resize back to original for display
        if resize_factor != 1.0 and original_size is not None:
            result_frame = cv2.resize(result_frame, (original_size[1], original_size[0]))
        
        # Convert to RGB for Streamlit display
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_display.image(result_frame_rgb, caption="Detection in progress", use_container_width=True)
        
        # Increment frame index
        frame_idx += 1
        
        # Add a small delay to control frame rate and prevent UI freezing
        time.sleep(0.001)
    
    # Release resources
    cap.release()
    st.session_state.processing = False
    
    if frame_idx > 0:
        st.success(f"Video processing complete! Processed {st.session_state.processed_frames} frames.")

# Handle button clicks
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if start_button:
        st.session_state.processing = True
        process_video(video_path, model, conf_threshold)
    
    if stop_button:
        st.session_state.processing = False
else:
    video_display.info("Please upload a video file to begin")

# Footer
st.markdown("---")
st.markdown('<div class="footer">© 2024 Fire and Smoke Detection System | Powered by YOLOv8</div>', unsafe_allow_html=True) 