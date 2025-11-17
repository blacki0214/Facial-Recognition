import tensorflow as tf
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance
    
    def _initialize_models(self):
        """Load all models once"""
        print("🔄 Loading models...")
        
        # Load emotion model
        self.emotion_model = tf.keras.models.load_model(settings.EMOTION_MODEL_PATH)
        print(f"✅ Emotion model loaded")
        print(f"   Input shape: {self.emotion_model.input_shape}")
        print(f"   Expected: (None, 48, 48, 1)")
        
        # Load embedding model
        self.embedding_model = tf.keras.models.load_model(settings.EMBEDDING_MODEL_PATH)
        print(f"✅ Embedding model loaded")
        print(f"   Input shape: {self.embedding_model.input_shape}")
        print(f"   Expected: (None, 160, 160, 3)")
        
        # Load liveness model
        self.liveness_model = tf.keras.models.load_model(settings.LIVENESS_MODEL_PATH)
        print(f"✅ Liveness model loaded")
        print(f"   Input shape: {self.liveness_model.input_shape}")
        print(f"   Expected: (None, 160, 160, 3)")
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Face cascade loaded")
        
        print("🎉 All models loaded successfully!")
    
    def get_emotion_model(self):
        return self.emotion_model
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_liveness_model(self):
        return self.liveness_model
    
    def get_face_cascade(self):
        return self.face_cascade