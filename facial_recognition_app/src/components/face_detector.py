import cv2
import numpy as np
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.model_loader import ModelLoader
from utils.face_processor import FaceProcessor
from config import settings

class FaceDetector:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.face_cascade = self.model_loader.get_face_cascade()
        self.embedding_model = self.model_loader.get_embedding_model()
        self.liveness_model = self.model_loader.get_liveness_model()
        
        print(f"✅ FaceDetector initialized")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray
    
    def check_liveness(self, face):
        """Check if face is live (not spoofed)"""
        try:
            # Preprocess for liveness model (160x160 RGB)
            face_normalized = FaceProcessor.preprocess_for_liveness(face)
            
            # Predict
            prediction = self.liveness_model.predict(face_normalized, verbose=0)[0][0]
            
            # Model outputs: >= 0.5 = Real, < 0.5 = Fake
            is_live = prediction >= 0.5
            confidence = prediction if is_live else (1 - prediction)
            
            return is_live, float(confidence)
            
        except Exception as e:
            print(f"❌ Liveness detection error: {e}")
            traceback.print_exc()
            return False, 0.0
    
    def get_face_embedding(self, face):
        """Extract face embedding (224x224 RGB)"""
        face_processed = FaceProcessor.preprocess_for_embedding(face)
        embedding = self.embedding_model.predict(face_processed, verbose=0)[0]
        return embedding
    
    def verify_face(self, face):
        """Verify face against saved embeddings"""
        embedding = self.get_face_embedding(face)
        name, similarity = FaceProcessor.find_match(embedding)
        return name, similarity, embedding
    
    def register_face_multi_pose(self, faces_dict, user_id):
        """
        Register face with multiple poses
        faces_dict: {'front': face_img, 'left': face_img, 'right': face_img}
        """
        embeddings = []
        
        for pose, face in faces_dict.items():
            # Check liveness
            is_live, confidence = self.check_liveness(face)
            
            if not is_live:
                return False, f"Liveness check failed on {pose} pose (confidence: {confidence:.2f})"
            
            # Get embedding
            embedding = self.get_face_embedding(face)
            embeddings.append(embedding)
        
        # Save to database
        FaceProcessor.save_user_faces(user_id, faces_dict, embeddings)
        
        return True, f"Face registered successfully for {user_id}"