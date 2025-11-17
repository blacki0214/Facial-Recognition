import cv2
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

class FaceProcessor:
    @staticmethod
    def preprocess_for_emotion(face, img_size=settings.IMG_SIZE):
        """Preprocess face for emotion detection - 48x48 grayscale"""
        face_resized = cv2.resize(face, img_size)
        face_normalized = np.expand_dims(np.expand_dims(face_resized, -1), 0) / 255.0
        return face_normalized
    
    @staticmethod
    def preprocess_for_embedding(face, face_size=settings.FACE_SIZE):
        """Preprocess face for embedding extraction - 160x160 RGB"""
        print(f"DEBUG: Using face_size={face_size} from settings.FACE_SIZE={settings.FACE_SIZE}")
        
        # Resize to 160x160
        face_resized = cv2.resize(face, face_size)
        print(f"DEBUG: Resized face shape: {face_resized.shape}")
        
        # Convert to RGB if grayscale
        if len(face_resized.shape) == 2:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        print(f"DEBUG: RGB face shape: {face_rgb.shape}")
        
        # Normalize and add batch dimension
        face_normalized = face_rgb.astype("float32") / 255.0
        face_normalized = np.expand_dims(face_normalized, axis=0)
        
        print(f"DEBUG: Final normalized shape: {face_normalized.shape}")
        
        return face_normalized
    
    @staticmethod
    def preprocess_for_liveness(face, liveness_size=settings.LIVENESS_SIZE):
        """Preprocess face for liveness detection - 160x160 RGB"""
        # Resize to 160x160
        face_resized = cv2.resize(face, liveness_size)
        
        # Convert to RGB if grayscale
        if len(face_resized.shape) == 2:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and add batch dimension
        face_normalized = face_rgb.astype("float32") / 255.0
        face_normalized = np.expand_dims(face_normalized, axis=0)
        
        return face_normalized
    
    @staticmethod
    def save_user_faces(user_id, faces_dict, embeddings):
        """
        Save user faces and embeddings to structured database
        faces_dict: {'front': image, 'left': image, 'right': image}
        embeddings: list of embeddings for each pose
        """
        user_dir = settings.FACE_DB_DIR / user_id
        user_dir.mkdir(exist_ok=True)
        
        # Save images
        for pose, image in faces_dict.items():
            image_path = user_dir / f"{pose}.jpg"
            cv2.imwrite(str(image_path), image)
        
        # Save embeddings as numpy array
        embeddings_array = np.array(embeddings)
        embeddings_path = user_dir / "embeddings.npy"
        np.save(str(embeddings_path), embeddings_array)
        
        print(f"✅ Saved {user_id} to database")
    
    @staticmethod
    def load_face_database():
        """Load all registered faces and their embeddings"""
        face_db = {}
        
        if not settings.FACE_DB_DIR.exists():
            return face_db
        
        for user_dir in settings.FACE_DB_DIR.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                embeddings_path = user_dir / "embeddings.npy"
                
                if embeddings_path.exists():
                    embeddings = np.load(str(embeddings_path))
                    face_db[user_id] = embeddings
        
        return face_db
    
    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    @staticmethod
    def find_match(embedding, threshold=settings.VERIFICATION_THRESHOLD):
        """Find matching face from database using multiple embeddings"""
        face_db = FaceProcessor.load_face_database()
        
        if not face_db:
            return None, 0.0
        
        best_match = None
        best_similarity = 0
        
        for user_id, user_embeddings in face_db.items():
            similarities = []
            for stored_embedding in user_embeddings:
                sim = FaceProcessor.calculate_similarity(embedding, stored_embedding)
                similarities.append(sim)
            
            max_similarity = np.max(similarities)
            
            if max_similarity > best_similarity and max_similarity > threshold:
                best_similarity = max_similarity
                best_match = user_id
        
        return best_match, best_similarity
    
    @staticmethod
    def save_face_embedding(name, embedding):
        """Save face embedding to disk (legacy)"""
        embeddings_file = settings.EMBEDDINGS_DIR / "embeddings.pkl"
        
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
        else:
            embeddings_dict = {}
        
        embeddings_dict[name] = embedding
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    
    @staticmethod
    def load_embeddings():
        """Load all saved embeddings (legacy)"""
        embeddings_file = settings.EMBEDDINGS_DIR / "embeddings.pkl"
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                return pickle.load(f)
        return {}