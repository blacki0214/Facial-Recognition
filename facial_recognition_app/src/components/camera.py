import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from src.components.face_detector import FaceDetector
from src.components.emotion_detector import EmotionDetector

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_detector = EmotionDetector()
        self.mode = "verification"  # "verification" or "registration"
        self.registration_name = ""
        self.last_result = None
    
    def set_mode(self, mode, name=""):
        self.mode = mode
        self.registration_name = name
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect faces
        faces, gray = self.face_detector.detect_faces(img)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            
            if self.mode == "verification":
                # Verify face
                name, similarity, _ = self.face_detector.verify_face(face)
                
                # Detect emotion
                emotion, emotion_conf, _ = self.emotion_detector.detect_emotion(face)
                
                # Draw rectangle and text
                if name:
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{name} ({similarity:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                cv2.putText(img, f"{emotion} ({emotion_conf:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                self.last_result = {
                    "name": name,
                    "similarity": similarity,
                    "emotion": emotion,
                    "emotion_confidence": emotion_conf
                }
            
            elif self.mode == "registration" and self.registration_name:
                # Check liveness
                is_live, liveness_conf = self.face_detector.check_liveness(face)
                
                color = (0, 255, 0) if is_live else (0, 0, 255)
                label = f"Live: {is_live} ({liveness_conf:.2f})"
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def create_camera_component(mode="verification", registration_name=""):
    """Create camera component with face detection"""
    ctx = webrtc_streamer(
        key="face-detection",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.set_mode(mode, registration_name)
    
    return ctx

def webcam_capture(key="webcam"):
    """Capture image from webcam"""
    
    st.write("### 📷 Webcam Capture")
    
    # Create placeholder for camera feed
    img_placeholder = st.empty()
    
    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Camera", key=f"start_{key}")
    with col2:
        capture_button = st.button("📸 Capture", key=f"capture_{key}")
    with col3:
        stop_button = st.button("Stop Camera", key=f"stop_{key}")
    
    # Session state for camera
    if f'camera_active_{key}' not in st.session_state:
        st.session_state[f'camera_active_{key}'] = False
    if f'captured_image_{key}' not in st.session_state:
        st.session_state[f'captured_image_{key}'] = None
    
    if start_button:
        st.session_state[f'camera_active_{key}'] = True
    
    if stop_button:
        st.session_state[f'camera_active_{key}'] = False
    
    # Camera feed
    if st.session_state[f'camera_active_{key}']:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your camera permissions.")
            st.session_state[f'camera_active_{key}'] = False
            return None
        
        ret, frame = cap.read()
        
        if ret:
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Capture image
            if capture_button:
                st.session_state[f'captured_image_{key}'] = frame.copy()
                st.success("✅ Image captured!")
                st.session_state[f'camera_active_{key}'] = False
        
        cap.release()
    
    # Display captured image
    if st.session_state[f'captured_image_{key}'] is not None:
        st.write("### Captured Image")
        captured_rgb = cv2.cvtColor(st.session_state[f'captured_image_{key}'], cv2.COLOR_BGR2RGB)
        st.image(captured_rgb, channels="RGB", use_container_width=True)
        
        return st.session_state[f'captured_image_{key}']
    
    return None


def webcam_stream_with_detection(process_func, title="Live Detection", key="stream"):
    """Stream webcam with live detection"""
    
    st.write(f"### {title}")
    
    # Create placeholders
    img_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Stream", key=f"start_{key}")
    with col2:
        stop_button = st.button("⏹️ Stop Stream", key=f"stop_{key}")
    
    # Session state
    if f'stream_active_{key}' not in st.session_state:
        st.session_state[f'stream_active_{key}'] = False
    
    if start_button:
        st.session_state[f'stream_active_{key}'] = True
    
    if stop_button:
        st.session_state[f'stream_active_{key}'] = False
    
    # Stream
    if st.session_state[f'stream_active_{key}']:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera.")
            st.session_state[f'stream_active_{key}'] = False
            return
        
        # Process one frame at a time
        ret, frame = cap.read()
        
        if ret:
            # Process frame
            processed_frame, result_info = process_func(frame)
            
            # Display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if result_info:
                info_placeholder.info(result_info)
            
            # Small delay to allow streamlit to update
            time.sleep(0.1)
        
        cap.release()
        
        # Auto-rerun to continue stream
        if st.session_state[f'stream_active_{key}']:
            st.rerun()