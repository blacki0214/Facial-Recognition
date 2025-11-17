import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FACE_DB_DIR = DATA_DIR / "face_db"

# Model paths
EMOTION_MODEL_PATH = MODELS_DIR / "emotion_detector.keras"
LIVENESS_MODEL_PATH = MODELS_DIR / "liveness_detector_zalo.keras"
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding_model.keras"

# Model settings
FACE_SIZE = (160, 160)  # For face embedding
IMG_SIZE = (48, 48)  # For emotion detection
LIVENESS_SIZE = (160, 160)  # For liveness detection
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Detection settings
CONFIDENCE_THRESHOLD = 0.6
VERIFICATION_THRESHOLD = 0.6  # Cosine similarity threshold
LIVENESS_THRESHOLD = 0.5  # Model outputs >= 0.5 = Real

# Registration poses
REGISTRATION_POSES = ["front", "left", "right"]

# Create directories if they don't exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FACE_DB_DIR, exist_ok=True)