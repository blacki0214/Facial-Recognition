import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import time
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta

# Force clear all Streamlit caches
st.cache_data.clear()
st.cache_resource.clear()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.face_detector import FaceDetector
from components.emotion_detector import EmotionDetector
from utils.face_processor import FaceProcessor
from config import settings

st.set_page_config(
    page_title="Facial Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
    }
    
    .step {
        text-align: center;
        flex: 1;
    }
    
    .step-circle {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: #e0e0e0;
        color: #666;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
    }
    
    .step-circle.active {
        background: #4285f4;
        color: white;
    }
    
    .step-circle.completed {
        background: #34a853;
        color: white;
    }
    
    /* Camera frame */
    .camera-frame {
        border: 3px solid #4285f4;
        border-radius: 12px;
        padding: 10px;
        background: white;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Emotion chart */
    .emotion-bar {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    
    .emotion-label {
        width: 100px;
        font-weight: 500;
    }
    
    .emotion-progress {
        flex: 1;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0 10px;
    }
    
    .emotion-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    
    .emotion-percent {
        width: 50px;
        text-align: right;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    return {
        'face_detector': FaceDetector(),
        'emotion_detector': EmotionDetector()
    }

# Initialize models
models = load_models()

# Initialize session state
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = defaultdict(lambda: {
        'name': 'Unknown',
        'emotion': 'Neutral',
        'liveness': False,
        'last_seen': time.time()
    })

if 'registration_state' not in st.session_state:
    st.session_state.registration_state = {
        'user_id': '',
        'current_pose': 0,
        'captured_faces': {},
        'captured_images': {}
    }

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

def main():
    # Sidebar navigation
    st.sidebar.markdown("### 🎭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📝 Register Face", "📊 Real-Time Tracking", 
         "🧪 Test Liveness", "😊 Test Emotion", "📁 Data Management"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    face_db = FaceProcessor.load_face_database()
    st.sidebar.metric("👥 Registered Users", len(face_db))
    
    
    # Main content
    if page == "🏠 Home":
        home_page()
    elif page == "📝 Register Face":
        register_face_page()
    elif page == "📊 Real-Time Tracking":
        realtime_tracking_page()
    elif page == "🧪 Test Liveness":
        test_liveness_page()
    elif page == "😊 Test Emotion":
        test_emotion_page()
    else:
        database_page()

def register_face_page():
    st.title("📝 Face Registration")
    st.markdown("##### Capture one front-facing photo")
    st.markdown("---")
    
    # User ID input
    if not st.session_state.registration_state['user_id']:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            user_id = st.text_input("📝 Enter Name or ID:", placeholder="e.g., John Doe")
            
            if st.button("▶️ Start Registration", use_container_width=True):
                if user_id:
                    st.session_state.registration_state['user_id'] = user_id
                    st.session_state.registration_state['current_pose'] = 0
                    st.session_state.registration_state['captured_faces'] = {}
                    st.session_state.registration_state['captured_images'] = {}
                    st.rerun()
                else:
                    st.error("Please enter a name or ID")
        return
    
    # Progress indicator
    user_id = st.session_state.registration_state['user_id']
    current_pose_idx = st.session_state.registration_state['current_pose']
    
    if current_pose_idx == 0:
        # Instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="camera-frame">
                <h4 style="text-align:center; color:#4285f4;">
                    👁️ Look straight at the camera
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            camera_input = st.camera_input("Look at camera", key="camera_front", label_visibility="collapsed")
        
        with col2:
            st.markdown("### 📋 Instructions")
            st.info(f"""
            **Capture Front View**
            
            👁️ Look straight at the camera
            
            ✓ Ensure good lighting  
            ✓ Face clearly visible  
            ✓ No mask
            """)
        
        if camera_input:
            # Process image
            image = Image.open(camera_input)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect face
            faces, gray = models['face_detector'].detect_faces(img_bgr)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]
                face_color = img_bgr[y:y+h, x:x+w]
                
                # Check liveness
                is_live, liveness_conf = models['face_detector'].check_liveness(face)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Draw rectangle
                    img_display = img_bgr.copy()
                    color = (0, 255, 0) if is_live else (0, 0, 255)
                    cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 3)
                    
                    status = "✅ Real" if is_live else "❌ Fake"
                    cv2.putText(img_display, status, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                
                with col2:
                    if is_live:
                        st.success("### ✅ Real Face")
                        st.metric("Confidence", f"{liveness_conf:.1%}")
                        
                        if st.button("✅ Confirm Photo", use_container_width=True):
                            st.session_state.registration_state['captured_faces']['front'] = face
                            st.session_state.registration_state['captured_images']['front'] = face_color
                            st.session_state.registration_state['current_pose'] = 1
                            st.success("Photo captured!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("### ❌ Spoof Detected")
                        st.metric("Confidence", f"{liveness_conf:.1%}")
                        st.warning("Please try again with a real face")
                        
            elif len(faces) > 1:
                st.error("⚠️ Multiple faces detected. Please ensure only 1 person is visible.")
            else:
                st.warning("⚠️ No face detected. Please try again.")
    
    else:
        # Photo captured
        st.success("### ✅ Photo captured!")
        
        # Display captured image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if 'front' in st.session_state.registration_state['captured_images']:
                img = st.session_state.registration_state['captured_images']['front']
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Front View", use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to Database", use_container_width=True):
                with st.spinner("Saving..."):
                    # Save single front face
                    face = st.session_state.registration_state['captured_faces']['front']
                    face_img = st.session_state.registration_state['captured_images']['front']
                    
                    # Save using FaceProcessor
                    user_dir = settings.FACE_DB_DIR / user_id
                    user_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save image
                    cv2.imwrite(str(user_dir / "front.jpg"), face_img)
                    
                    # Generate and save embedding using FaceProcessor
                    try:
                        # Resize COLOR image for embedding (embedding model uses RGB)
                        face_resized = cv2.resize(face_img, (160, 160))
                        
                        # Convert BGR to RGB
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        
                        # Normalize
                        face_normalized = face_rgb / 255.0
                        
                        # Add batch dimension
                        face_input = np.expand_dims(face_normalized, axis=0)
                        
                        # Generate embedding
                        embedding = models['face_detector'].embedding_model.predict(face_input, verbose=0)[0]
                        
                        # Save embedding
                        embeddings = [embedding]
                        np.save(str(user_dir / "embeddings.npy"), embeddings)
                        
                        st.success(f"✅ Face registered successfully for {user_id}")
                        st.balloons()
                        time.sleep(2)
                        st.session_state.registration_state = {
                            'user_id': '',
                            'current_pose': 0,
                            'captured_faces': {},
                            'captured_images': {}
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate face embedding: {str(e)}")
        
        with col2:
            if st.button("🔄 Retake", use_container_width=True):
                st.session_state.registration_state = {
                    'user_id': '',
                    'current_pose': 0,
                    'captured_faces': {},
                    'captured_images': {}
                }
                st.rerun()

def home_page():
    st.title("👤 Facial Recognition System")
    st.markdown("##### AI-Powered Intelligent Recognition System")
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📝 Face Registration</h3>
            <p>Capture 1 face image:</p>
            <ul>
                <li>👁️ Front view</li>
            </ul>
            <p><small>Automatic liveness detection included</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Real-Time Tracking</h3>
            <p>Monitoring features:</p>
            <ul>
                <li>✅ Identity recognition</li>
                <li>😊 Emotion detection</li>
                <li>🛡️ Liveness check</li>
            </ul>
            <p><small>Live information overlay on video</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 Feature Testing</h3>
            <p>Testing tools:</p>
            <ul>
                <li>🛡️ Liveness Detection</li>
                <li>😊 Emotion Detection</li>
                <li>📈 Analytics Dashboard</li>
            </ul>
            <p><small>Evaluate model accuracy</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System statistics
    face_db = FaceProcessor.load_face_database()
    
    

def register_face_page():
    st.title("📝 Face Registration")
    st.markdown("##### Capture one front-facing photo")
    st.markdown("---")
    
    # User ID input
    if not st.session_state.registration_state['user_id']:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            user_id = st.text_input("📝 Enter Name or ID:", placeholder="e.g., John Doe")
            
            if st.button("▶️ Start Registration", use_container_width=True):
                if user_id:
                    st.session_state.registration_state['user_id'] = user_id
                    st.session_state.registration_state['current_pose'] = 0
                    st.session_state.registration_state['captured_faces'] = {}
                    st.session_state.registration_state['captured_images'] = {}
                    st.rerun()
                else:
                    st.error("Please enter a name or ID")
        return
    
    # Progress indicator
    user_id = st.session_state.registration_state['user_id']
    current_pose_idx = st.session_state.registration_state['current_pose']
    
    if current_pose_idx == 0:
        # Instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="camera-frame">
                <h4 style="text-align:center; color:#4285f4;">
                    👁️ Look straight at the camera
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            camera_input = st.camera_input("Look at camera", key="camera_front", label_visibility="collapsed")
        
        with col2:
            st.markdown("### 📋 Instructions")
            st.info(f"""
            **Capture Front View**
            
            👁️ Look straight at the camera
            
            ✓ Ensure good lighting  
            ✓ Face clearly visible  
            ✓ No mask
            """)
        
        if camera_input:
            # Process image
            image = Image.open(camera_input)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect face
            faces, gray = models['face_detector'].detect_faces(img_bgr)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]
                face_color = img_bgr[y:y+h, x:x+w]
                
                # Check liveness
                is_live, liveness_conf = models['face_detector'].check_liveness(face)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Draw rectangle
                    img_display = img_bgr.copy()
                    color = (0, 255, 0) if is_live else (0, 0, 255)
                    cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 3)
                    
                    status = "✅ Real" if is_live else "❌ Fake"
                    cv2.putText(img_display, status, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                
                with col2:
                    if is_live:
                        st.success("### ✅ Real Face")
                        st.metric("Confidence", f"{liveness_conf:.1%}")
                        
                        if st.button("✅ Confirm Photo", use_container_width=True):
                            st.session_state.registration_state['captured_faces']['front'] = face
                            st.session_state.registration_state['captured_images']['front'] = face_color
                            st.session_state.registration_state['current_pose'] = 1
                            st.success("Photo captured!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("### ❌ Spoof Detected")
                        st.metric("Confidence", f"{liveness_conf:.1%}")
                        st.warning("Please try again with a real face")
                        
            elif len(faces) > 1:
                st.error("⚠️ Multiple faces detected. Please ensure only 1 person is visible.")
            else:
                st.warning("⚠️ No face detected. Please try again.")
    
    else:
        # Photo captured
        st.success("### ✅ Photo captured!")
        
        # Display captured image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if 'front' in st.session_state.registration_state['captured_images']:
                img = st.session_state.registration_state['captured_images']['front']
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Front View", use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to Database", use_container_width=True):
                with st.spinner("Saving..."):
                    # Save single front face
                    face = st.session_state.registration_state['captured_faces']['front']
                    face_img = st.session_state.registration_state['captured_images']['front']
                    
                    # Save using FaceProcessor
                    user_dir = settings.FACE_DB_DIR / user_id
                    user_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save image
                    cv2.imwrite(str(user_dir / "front.jpg"), face_img)
                    
                    # Generate and save embedding using FaceProcessor
                    try:
                        # Resize COLOR image for embedding (embedding model uses RGB)
                        face_resized = cv2.resize(face_img, (160, 160))
                        
                        # Convert BGR to RGB
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        
                        # Normalize
                        face_normalized = face_rgb / 255.0
                        
                        # Add batch dimension
                        face_input = np.expand_dims(face_normalized, axis=0)
                        
                        # Generate embedding
                        embedding = models['face_detector'].embedding_model.predict(face_input, verbose=0)[0]
                        
                        # Save embedding
                        embeddings = [embedding]
                        np.save(str(user_dir / "embeddings.npy"), embeddings)
                        
                        st.success(f"✅ Face registered successfully for {user_id}")
                        st.balloons()
                        time.sleep(2)
                        st.session_state.registration_state = {
                            'user_id': '',
                            'current_pose': 0,
                            'captured_faces': {},
                            'captured_images': {}
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate face embedding: {str(e)}")
        
        with col2:
            if st.button("🔄 Retake", use_container_width=True):
                st.session_state.registration_state = {
                    'user_id': '',
                    'current_pose': 0,
                    'captured_faces': {},
                    'captured_images': {}
                }
                st.rerun()

def realtime_tracking_page():
    st.title("📊 Real-Time Tracking")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📊 Live Emotion Chart")
        emotion_chart = st.empty()
        
        st.markdown("---")
        
        st.markdown("### 🔍 Detection List")
        detections_list = st.empty()
    
    with col1:
        run_camera = st.checkbox("▶️ Start Camera", value=False)
        FRAME_WINDOW = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera access error")
                    break
                
                # Process frame
                processed_frame, detections = process_tracking_frame(frame)
                
                # Display
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Update emotion chart
                if detections:
                    latest_emotion = detections[0].get('emotion_probs', {})
                    if latest_emotion:
                        with emotion_chart.container():
                            for emotion, prob in sorted(latest_emotion.items(), key=lambda x: x[1], reverse=True):
                                emoji = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Surprise": "😲", 
                                        "Fear": "😨", "Neutral": "😐", "Disgust": "🤢"}.get(emotion, "")
                                
                                color = {"Happy": "#43e97b", "Sad": "#4285f4", "Angry": "#f5576c",
                                        "Surprise": "#ffa726", "Fear": "#9c27b0", "Neutral": "#9e9e9e",
                                        "Disgust": "#8bc34a"}.get(emotion, "#999")
                                
                                st.markdown(f"""
                                <div class="emotion-bar">
                                    <div class="emotion-label">{emoji} {emotion}</div>
                                    <div class="emotion-progress">
                                        <div class="emotion-fill" style="width: {prob*100}%; background: {color};"></div>
                                    </div>
                                    <div class="emotion-percent">{prob:.0%}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Update detections
                with detections_list.container():
                    for det in detections[:5]:
                        if det['name'] != 'Unknown':
                            st.success(f"✅ **{det['name']}** ({det['similarity']:.0%})")
                        else:
                            st.info(f"❓ Unknown - {det['emotion']}")
                
                # Check if still running
                run_camera = st.session_state.get('run_camera', run_camera)
            
            cap.release()

def process_tracking_frame(frame):
    """Process frame with tracking"""
    faces, gray = models['face_detector'].detect_faces(frame)
    
    detections = []
    
    for idx, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]
        
        # Check liveness
        is_live, liveness_conf = models['face_detector'].check_liveness(face)
        
        if is_live and liveness_conf > settings.LIVENESS_THRESHOLD:
            # Verify face
            name, similarity, _ = models['face_detector'].verify_face(face)
            
            # Detect emotion
            emotion, emotion_conf, emotion_probs = models['emotion_detector'].detect_emotion(face)
            
            # Draw info
            color = (0, 255, 0) if name else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Background
            cv2.rectangle(frame, (x, y-70), (x+w, y), (0, 0, 0), -1)
            
            # Text
            label = name if name else "Unknown"
            cv2.putText(frame, label, (x+5, y-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if name:
                cv2.putText(frame, f"{similarity:.0%}", (x+5, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(frame, f"{emotion} ({emotion_conf:.0%})", (x+5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            detections.append({
                'name': name if name else 'Unknown',
                'similarity': similarity if name else 0,
                'emotion': emotion,
                'emotion_probs': emotion_probs,
                'liveness': True
            })
        else:
            # Spoof detected
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, "SPOOF!", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, detections

def test_liveness_page():
    st.title("🧪 Liveness Detection Test")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_input = st.camera_input("📸 Take a photo to test")
    
    with col2:
        st.info("""
        ### 📋 Instructions
        
        Test if the face is real or fake
        
        ✓ Real face  
        ✗ Printed photo/screen
        """)
    
    if camera_input:
        image = Image.open(camera_input)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        faces, gray = models['face_detector'].detect_faces(img_bgr)
        
        if faces is not None and len(faces) > 0:
            for x, y, w, h in faces:
                face = gray[y:y+h, x:x+w]
                
                is_live, confidence = models['face_detector'].check_liveness(face)
                
                # Draw
                color = (0, 255, 0) if is_live else (0, 0, 255)
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)
                
                label = "REAL" if is_live else "FAKE"
                cv2.putText(img_bgr, f"{label} ({confidence:.0%})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
            
            with col2:
                if is_live:
                    st.success("### ✅ REAL FACE")
                else:
                    st.error("### ❌ SPOOF DETECTED")
                
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Progress bar
                st.progress(confidence)
        else:
            st.warning("⚠️ No face detected")

def test_emotion_page():
    st.title("😊 Emotion Detection Test")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_input = st.camera_input("📸 Take a photo to detect emotion")
    
    with col2:
        st.info("""
        ### 📋 Emotions Detected
        
        😊 Happy  
        😢 Sad  
        😠 Angry  
        😲 Surprise  
        😨 Fear  
        😐 Neutral  
        🤢 Disgust
        """)
    
    if camera_input:
        image = Image.open(camera_input)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        faces, gray = models['face_detector'].detect_faces(img_bgr)
        
        if faces is not None and len(faces) > 0:
            for x, y, w, h in faces:
                face = gray[y:y+h, x:x+w]
                
                emotion, confidence, emotion_probs = models['emotion_detector'].detect_emotion(face)
                
                # Draw
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_bgr, f"{emotion} ({confidence:.0%})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
            
            with col2:
                emoji_map = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Surprise": "😲", 
                            "Fear": "😨", "Neutral": "😐", "Disgust": "🤢"}
                
                st.success(f"### {emoji_map.get(emotion, '')} {emotion}")
                st.metric("Confidence", f"{confidence:.1%}")
                
                st.markdown("### Real-Time Emotion Chart")
                for em, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
                    emoji = emoji_map.get(em, "")
                    color = {"Happy": "#43e97b", "Sad": "#4285f4", "Angry": "#f5576c",
                            "Surprise": "#ffa726", "Fear": "#9c27b0", "Neutral": "#9e9e9e",
                            "Disgust": "#8bc34a"}.get(em, "#999")
                    
                    st.markdown(f"""
                    <div class="emotion-bar">
                        <div class="emotion-label">{emoji} {em}</div>
                        <div class="emotion-progress">
                            <div class="emotion-fill" style="width: {prob*100}%; background: {color};"></div>
                        </div>
                        <div class="emotion-percent">{prob:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No face detected")

def database_page():
    st.title("📁 Face Data Management")
    st.markdown("##### Manage registered face database")
    st.markdown("---")
    
    face_db = FaceProcessor.load_face_database()
    
    if face_db:
        # Summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Total Faces</h4>
                <div style="font-size: 36px; font-weight: bold; color: #4285f4;">{len(face_db)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            active = sum(1 for v in face_db.values() if len(v) >= 1)
            st.markdown(f"""
            <div class="metric-card">
                <h4>✅ Active Users</h4>
                <div style="font-size: 36px; font-weight: bold; color: #34a853;">{active}</div>
            </div>
            """, unsafe_allow_html=True)
        
        
        
        # User list
        for idx, user_id in enumerate(face_db.keys()):
            with st.expander(f"#{idx+1:03d} - 👤 {user_id}", expanded=False):
                user_dir = settings.FACE_DB_DIR / user_id
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    img_path = user_dir / "front.jpg"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption="Front View", use_container_width=True)
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_{user_id}"):
                        import shutil
                        shutil.rmtree(user_dir)
                        st.success(f"Deleted {user_id}")
                        st.rerun()
    else:
        st.info("📭 No users registered yet")
        if st.button("➕ Register new face"):
            st.session_state.current_page = "register"
            st.rerun()

if __name__ == "__main__":
    main()