import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
import random
import keras

class FacialRecognitionSystem:
    def __init__(self):
        # Load models
        self.emotion_model = tf.keras.models.load_model("models/emotion_detector.keras")
        self.face_model = tf.keras.models.load_model("models/embedding_model.keras")
        
        try:
            self.liveness_model = tf.keras.models.load_model("models/liveness_detector_zalo.keras")
        except:
            print("Warning: Liveness model not found. Using default checks.")
            self.liveness_model = None
        
        # Load registered faces database
        self.load_database()
        
        # Emotion labels
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Liveness challenge system
        self.liveness_challenge = None
        self.challenge_frame_count = 0
        self.challenge_max_frames = 90  # 3 seconds at 30fps
        self.challenge_completed = False
        self.challenges = ["blink", "smile"]
        self.blink_counter = 0
        self.last_blink_state = False
        self.ear_history = []
        
    def load_database(self):
        """Load registered faces from pickle file"""
        db_path = "data/face_database.pkl"
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.face_db = pickle.load(f)
        else:
            self.face_db = {}  # {name: embedding_vector}
    
    def save_database(self):
        """Save registered faces to pickle file"""
        os.makedirs("data", exist_ok=True)
        with open("data/face_database.pkl", 'wb') as f:
            pickle.dump(self.face_db, f)
    
    def register_face(self, name, frame):
        """Register a new face with name"""
        # Extract face embedding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, "No face detected"
        
        if len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only one face in frame"
        
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        face_arr = np.expand_dims(face_resized / 255.0, 0)
        
        # Get embedding
        embedding = self.face_model.predict(face_arr, verbose=0).squeeze()
        
        # Save to database
        self.face_db[name] = embedding
        self.save_database()
        
        return True, f"Face registered successfully for {name}"
    
    def recognize_face(self, frame):
        """Recognize face and return name + confidence"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            face_arr = np.expand_dims(face_resized / 255.0, 0)
            
            # Get embedding
            embedding = self.face_model.predict(face_arr, verbose=0).squeeze()
            
            # Compare with database
            min_dist = float('inf')
            matched_name = "Unknown"
            
            for name, db_embedding in self.face_db.items():
                dist = np.linalg.norm(embedding - db_embedding)
                if dist < min_dist:
                    min_dist = dist
                    matched_name = name
            
            # Threshold for recognition
            confidence = max(0, 1 - min_dist)
            if min_dist > 0.6:  # Adjust threshold as needed
                matched_name = "Unknown"
            
            results.append({
                'bbox': (x, y, w, h),
                'name': matched_name,
                'confidence': confidence
            })
        
        return results
    
    def detect_emotion(self, frame, bbox):
        """Detect emotion for given face bbox"""
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_arr = np.expand_dims(np.expand_dims(face_resized, -1), 0) / 255.0
        
        preds = self.emotion_model.predict(face_arr, verbose=0)[0]
        emotion = self.emotions[np.argmax(preds)]
        confidence = np.max(preds)
        
        return emotion, confidence
    
    def check_liveness(self, frame, bbox):
        """Check if face is real using liveness model"""
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        # Implement your liveness detection logic
        # Return True if real, False if fake
        return True  # Placeholder
    
    def run(self):
        """Main loop for real-time detection"""
        cap = cv2.VideoCapture(0)
        mode = "recognition"  # or "registration"
        
        print("Controls:")
        print("'q' - Quit")
        print("'r' - Registration mode")
        print("'d' - Detection mode")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if mode == "recognition":
                # Recognize faces
                faces = self.recognize_face(frame)
                
                for face_info in faces:
                    x, y, w, h = face_info['bbox']
                    name = face_info['name']
                    conf = face_info['confidence']
                    
                    # Detect emotion
                    emotion, emotion_conf = self.detect_emotion(frame, (x, y, w, h))
                    
                    # Draw bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display info
                    cv2.putText(frame, f"{name} ({conf:.2f})", 
                              (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Emotion: {emotion}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display mode
            cv2.putText(frame, f"Mode: {mode.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Facial Recognition System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                mode = "registration"
                name = input("Enter name to register: ")
                success, msg = self.register_face(name, frame)
                print(msg)
                mode = "recognition"
            elif key == ord('d'):
                mode = "recognition"
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FacialRecognitionSystem()
    system.run()