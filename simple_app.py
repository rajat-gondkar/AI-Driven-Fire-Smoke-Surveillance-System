import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image

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
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔥 Fire and Smoke Detection</h1>', unsafe_allow_html=True)

# Load the model at startup
@st.cache_resource
def load_model():
    return YOLO("runs/detect/fire_smoke_detection2/weights/best.pt")

model = load_model()

# Initialize variables
if 'fire_detected' not in st.session_state:
    st.session_state.fire_detected = False
if 'smoke_detected' not in st.session_state:
    st.session_state.smoke_detected = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    # Confidence threshold
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Process button
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")
    
    # Display alert status
    if st.session_state.fire_detected:
        st.markdown('<div class="alert-fire"><strong>⚠️ FIRE DETECTED!</strong></div>', unsafe_allow_html=True)
    
    if st.session_state.smoke_detected:
        st.markdown('<div class="alert-smoke"><strong>⚠️ SMOKE DETECTED!</strong></div>', unsafe_allow_html=True)

with col2:
    # Display area for video
    video_display = st.empty()

# Process video function
def process_video(video_path, conf_threshold):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Reset detection status
    st.session_state.fire_detected = False
    st.session_state.smoke_detected = False
    
    # Process the video
    while cap.isOpened() and st.session_state.processing:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame, conf=conf_threshold)
        
        # Process results
        result_frame = results[0].plot()
        
        # Check for fire and smoke
        for detection in results[0].boxes.data.tolist():
            class_id = int(detection[5])
            confidence = detection[4]
            
            if class_id == 0 and confidence > conf_threshold:  # Fire class
                st.session_state.fire_detected = True
            elif class_id == 1 and confidence > conf_threshold:  # Smoke class
                st.session_state.smoke_detected = True
        
        # Display the frame
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        video_display.image(result_frame_rgb, caption="Detection in progress", use_container_width=True)
        
        # Add a small delay to control frame rate
        time.sleep(0.01)
    
    # Release resources
    cap.release()
    st.session_state.processing = False

# Handle button clicks
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    if start_button:
        st.session_state.processing = True
        process_video(video_path, conf_threshold)
    
    if stop_button:
        st.session_state.processing = False
else:
    video_display.info("Please upload a video file to begin") 