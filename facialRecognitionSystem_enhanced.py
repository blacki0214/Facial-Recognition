import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
import random

class FacialRecognitionSystem:
    def __init__(self):
        print("Loading models...")
        # Load models
        self.emotion_model = tf.keras.models.load_model("models/emotion_detector.keras")
        self.face_model = tf.keras.models.load_model("models/embedding_model.keras")
        
        try:
            self.liveness_model = tf.keras.models.load_model("models/liveness_detector_zalo.keras")
            print("✓ Liveness model loaded")
        except:
            print("⚠ Warning: Liveness model not found. Using basic liveness checks.")
            self.liveness_model = None
        
        # Load registered faces database
        self.load_database()
        
        # Emotion labels
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Eye cascade for blink detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        
        # Liveness challenge system
        self.liveness_mode = False
        self.liveness_challenge = None
        self.challenge_frame_count = 0
        self.challenge_max_frames = 120  # 4 seconds at 30fps
        self.challenge_completed = False
        self.challenges = ["blink", "smile", "neutral"]
        self.blink_counter = 0
        self.eye_detected_frames = 0
        self.eye_not_detected_frames = 0
        self.last_eye_state = True
        
        print("✓ System initialized successfully!\n")
    
    def load_database(self):
        """Load registered faces from pickle file"""
        db_path = "data/face_database.pkl"
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.face_db = pickle.load(f)
            print(f"✓ Loaded {len(self.face_db)} registered faces")
        else:
            self.face_db = {}  # {name: embedding_vector}
            print("✓ Created new face database")
    
    def save_database(self):
        """Save registered faces to pickle file"""
        os.makedirs("data", exist_ok=True)
        with open("data/face_database.pkl", 'wb') as f:
            pickle.dump(self.face_db, f)
    
    def detect_eyes(self, frame, face_bbox):
        """Detect eyes in face region"""
        x, y, w, h = face_bbox
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        return len(eyes) >= 2  # At least 2 eyes detected
    
    def detect_blink(self, frame, face_bbox):
        """Simple blink detection using eye cascade"""
        eyes_detected = self.detect_eyes(frame, face_bbox)
        
        # Eyes disappeared (closed) after being detected (open)
        if self.last_eye_state and not eyes_detected:
            self.eye_not_detected_frames += 1
        elif not self.last_eye_state and eyes_detected:
            # Blink completed (eyes reopened)
            if self.eye_not_detected_frames >= 2:  # Eyes were closed for at least 2 frames
                self.blink_counter += 1
                print(f"  Blink detected! Count: {self.blink_counter}")
            self.eye_not_detected_frames = 0
        
        self.last_eye_state = eyes_detected
        return self.blink_counter
    
    def start_liveness_challenge(self):
        """Start a random liveness challenge"""
        self.liveness_challenge = random.choice(self.challenges)
        self.challenge_frame_count = 0
        self.challenge_completed = False
        self.blink_counter = 0
        self.eye_not_detected_frames = 0
        self.last_eye_state = True
        print(f"\n{'='*50}")
        print(f"🔍 LIVENESS CHALLENGE: {self.get_challenge_text()}")
        print(f"{'='*50}\n")
    
    def get_challenge_text(self):
        """Get display text for current challenge"""
        if self.liveness_challenge == "blink":
            return "BLINK YOUR EYES TWICE"
        elif self.liveness_challenge == "smile":
            return "SMILE FOR 2 SECONDS"
        elif self.liveness_challenge == "neutral":
            return "KEEP A NEUTRAL FACE FOR 2 SECONDS"
        return ""
    
    def check_challenge_completion(self, frame, bbox, emotion, emotion_conf):
        """Check if current challenge is completed"""
        if not self.liveness_challenge:
            return False
        
        self.challenge_frame_count += 1
        
        if self.liveness_challenge == "blink":
            blinks = self.detect_blink(frame, bbox)
            if blinks >= 2:
                print("✓ Blink challenge completed!")
                return True
        
        elif self.liveness_challenge == "smile":
            if emotion == "Happy" and emotion_conf > 0.6:
                # Need to smile for sustained frames
                if self.challenge_frame_count > 30:  # ~1 second
                    print("✓ Smile challenge completed!")
                    return True
        
        elif self.liveness_challenge == "neutral":
            if emotion == "Neutral" and emotion_conf > 0.5:
                if self.challenge_frame_count > 30:  # ~1 second
                    print("✓ Neutral challenge completed!")
                    return True
        
        # Timeout
        if self.challenge_frame_count > self.challenge_max_frames:
            print("✗ Challenge timeout! Try again.")
            return None  # Return None to indicate timeout
        
        return False
    
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
    
    def check_liveness_model(self, frame, bbox):
        """Check if face is real using liveness model"""
        if self.liveness_model is None:
            return True, 0.0
        
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        
        try:
            # Resize for liveness model (adjust size based on your model)
            face_resized = cv2.resize(face, (160, 160))
            face_arr = np.expand_dims(face_resized / 255.0, 0)
            
            # Predict
            pred = self.liveness_model.predict(face_arr, verbose=0)[0]
            
            # Assuming binary classification: [fake, real]
            is_real = pred[1] > 0.5 if len(pred) == 2 else pred[0] > 0.5
            confidence = pred[1] if len(pred) == 2 else pred[0]
            
            return is_real, confidence
        except Exception as e:
            print(f"Liveness model error: {e}")
            return True, 0.0
    
    def show_registered_faces(self):
        """Display list of registered faces"""
        print("\n" + "="*50)
        print("REGISTERED FACES")
        print("="*50)
        if len(self.face_db) == 0:
            print("No faces registered yet.")
        else:
            for idx, name in enumerate(self.face_db.keys(), 1):
                print(f"{idx}. {name}")
        print("="*50 + "\n")
    
    def delete_face(self, name):
        """Delete a registered face from database"""
        if name in self.face_db:
            del self.face_db[name]
            self.save_database()
            return True, f"Face '{name}' deleted successfully"
        return False, f"Face '{name}' not found in database"
    
    def run(self):
        """Main loop for real-time detection"""
        cap = cv2.VideoCapture(0)
        mode = "recognition"  # modes: "recognition", "registration", "liveness"
        registration_name = ""
        
        print("\n" + "="*60)
        print("  FACIAL RECOGNITION SYSTEM WITH EMOTION & LIVENESS")
        print("="*60)
        print("\n📋 CONTROLS:")
        print("  'q' - Quit")
        print("  'r' - Register new face")
        print("  'd' - Detection mode (default)")
        print("  'l' - Liveness challenge mode")
        print("  's' - Show registered faces")
        print("  'x' - Delete a registered face")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            display_frame = frame.copy()
            
            if mode == "recognition":
                # Recognize faces
                faces = self.recognize_face(frame)
                
                for face_info in faces:
                    x, y, w, h = face_info['bbox']
                    name = face_info['name']
                    conf = face_info['confidence']
                    
                    # Detect emotion
                    emotion, emotion_conf = self.detect_emotion(frame, (x, y, w, h))
                    
                    # Check liveness with model
                    is_real, liveness_conf = self.check_liveness_model(frame, (x, y, w, h))
                    
                    # Draw bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display info with background for better readability
                    y_offset = y - 15
                    
                    # Name and confidence
                    text = f"{name} ({conf:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x, y_offset - text_height - 5), 
                                (x + text_width, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(display_frame, text, (x, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Emotion
                    y_offset -= 30
                    text = f"Emotion: {emotion} ({emotion_conf:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(display_frame, (x, y_offset - text_height - 5), 
                                (x + text_width, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(display_frame, text, (x, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Liveness
                    if self.liveness_model is not None:
                        y_offset -= 25
                        liveness_text = "REAL" if is_real else "FAKE"
                        liveness_color = (0, 255, 0) if is_real else (0, 0, 255)
                        text = f"Liveness: {liveness_text} ({liveness_conf:.2f})"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(display_frame, (x, y_offset - text_height - 5), 
                                    (x + text_width, y_offset + 5), (0, 0, 0), -1)
                        cv2.putText(display_frame, text, (x, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, liveness_color, 2)
            
            elif mode == "liveness":
                # Liveness challenge mode
                faces = self.recognize_face(frame)
                
                if len(faces) > 0:
                    face_info = faces[0]
                    x, y, w, h = face_info['bbox']
                    
                    # Start challenge if not started
                    if not self.liveness_challenge:
                        self.start_liveness_challenge()
                    
                    # Detect emotion for challenge
                    emotion, emotion_conf = self.detect_emotion(frame, (x, y, w, h))
                    
                    # Check challenge completion
                    result = self.check_challenge_completion(frame, (x, y, w, h), emotion, emotion_conf)
                    
                    if result is True:
                        self.challenge_completed = True
                        cv2.putText(display_frame, "CHALLENGE PASSED!", 
                                  (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        print("✓✓✓ LIVENESS VERIFIED ✓✓✓\n")
                        # Auto-return to recognition after 2 seconds
                        cv2.imshow("Facial Recognition System", display_frame)
                        cv2.waitKey(2000)
                        mode = "recognition"
                        self.liveness_challenge = None
                        continue
                    elif result is None:  # Timeout
                        mode = "recognition"
                        self.liveness_challenge = None
                        continue
                    
                    # Draw challenge instructions
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 165, 0), 3)
                    
                    # Challenge text with background
                    challenge_text = self.get_challenge_text()
                    (text_width, text_height), _ = cv2.getTextSize(challenge_text, 
                                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                    cv2.rectangle(display_frame, (10, 50), (text_width + 30, 100), (255, 165, 0), -1)
                    cv2.putText(display_frame, challenge_text, (20, 85), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    
                    # Progress bar
                    progress = min(1.0, self.challenge_frame_count / self.challenge_max_frames)
                    bar_width = 400
                    bar_height = 20
                    cv2.rectangle(display_frame, (20, 120), (20 + bar_width, 120 + bar_height), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (20, 120), 
                                (20 + int(bar_width * progress), 120 + bar_height), (0, 255, 0), -1)
                else:
                    # No face detected in liveness mode
                    cv2.putText(display_frame, "NO FACE DETECTED", (50, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Display mode and instructions
            mode_color = (0, 255, 0) if mode == "recognition" else (255, 165, 0) if mode == "liveness" else (255, 0, 0)
            cv2.rectangle(display_frame, (5, 5), (300, 45), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            
            cv2.imshow("Facial Recognition System", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            
            elif key == ord('r'):
                print("\n" + "="*50)
                print("FACE REGISTRATION")
                print("="*50)
                name = input("Enter name to register (or 'cancel' to abort): ")
                if name.lower() != 'cancel' and name.strip():
                    ret, frame = cap.read()
                    if ret:
                        success, msg = self.register_face(name.strip(), frame)
                        print(msg)
                        if success:
                            print(f"Total registered faces: {len(self.face_db)}")
                    else:
                        print("Failed to capture frame")
                print("="*50 + "\n")
            
            elif key == ord('d'):
                mode = "recognition"
                self.liveness_challenge = None
                print("→ Switched to DETECTION mode\n")
            
            elif key == ord('l'):
                mode = "liveness"
                self.liveness_challenge = None
                print("→ Switched to LIVENESS CHALLENGE mode\n")
            
            elif key == ord('s'):
                self.show_registered_faces()
            
            elif key == ord('x'):
                self.show_registered_faces()
                name = input("Enter name to delete (or 'cancel' to abort): ")
                if name.lower() != 'cancel' and name.strip():
                    success, msg = self.delete_face(name.strip())
                    print(msg)
        
        cap.release()
        cv2.destroyAllWindows()
        print("System terminated.\n")

if __name__ == "__main__":
    try:
        system = FacialRecognitionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
