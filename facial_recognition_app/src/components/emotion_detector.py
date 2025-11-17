import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.model_loader import ModelLoader
from utils.face_processor import FaceProcessor
from config import settings

class EmotionDetector:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.emotion_model = self.model_loader.get_emotion_model()
        self.emotion_labels = settings.EMOTION_LABELS
    
    def detect_emotion(self, face):
        """Detect emotion from face"""
        face_processed = FaceProcessor.preprocess_for_emotion(face)
        predictions = self.emotion_model.predict(face_processed, verbose=0)[0]
        
        emotion_idx = np.argmax(predictions)
        emotion = self.emotion_labels[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Get all emotions with probabilities
        emotion_probs = {
            label: float(prob) 
            for label, prob in zip(self.emotion_labels, predictions)
        }
        
        return emotion, confidence, emotion_probs