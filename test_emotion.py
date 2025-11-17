import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("./models/emotion_model.keras")
img_size = 224

# Emotion classes (ensure same order as your training generator)
class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face):
    face = cv2.resize(face, (img_size, img_size))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Predict emotion
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_input = preprocess_face(face_rgb)

        preds = model.predict(face_input)[0]
        emotion = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Label
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
