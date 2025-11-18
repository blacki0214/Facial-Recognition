# Facial Recognition System

A comprehensive facial recognition system with emotion detection and liveness detection capabilities, built with TensorFlow/Keras and OpenCV. The system includes both a standalone Python application and a web-based Streamlit interface.

## 🌟 Features

- **Face Recognition & Verification**: Identify and verify individuals using deep learning embeddings
- **Emotion Detection**: Real-time detection of 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Liveness Detection**: Anti-spoofing system with challenge-response mechanism
- **Interactive Challenges**: Blink detection, smile detection, and expression verification
- **Web Interface**: Modern Streamlit-based UI for easy interaction
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Face Database Management**: Register and manage known faces

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models](#models)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.15.0
- CUDA-compatible GPU (optional, for faster inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/blacki0214/Facial-Recognition.git
cd Facial-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download or train the models (models should be placed in `models/` directory):
   - `embedding_model.keras` - Face embedding model
   - `emotion_detector.keras` - Emotion detection model
   - `liveness_detector_zalo.keras` - Liveness detection model

## 📁 Project Structure

```
facial_recognition/
├── data/                                    # Data directory
│   ├── classification_data/                # Face classification datasets
│   │   ├── train_data/
│   │   ├── val_data/
│   │   └── test_data/
│   ├── verification_data/                  # Face verification datasets
│   ├── verification_pairs_test.txt         # Test pairs for verification
│   └── verification_pairs_val.txt          # Validation pairs
│
├── models/                                  # Trained models
│   ├── embedding_model.keras               # Face embedding model
│   ├── emotion_detector.keras              # Emotion detection model
│   ├── liveness_detector_zalo.keras        # Liveness detection model
│   ├── classification_face_recognition.h5  # Classification model
│   ├── softmax_model.keras
│   └── triplet_model.keras
│
├── notebooks/                               # Jupyter notebooks
│   ├── 01_data_processing.ipynb            # Data preprocessing
│   ├── 02_face_verfication.ipynb           # Face verification training
│   ├── 02b_compare_models.ipynb            # Model comparison
│   ├── 03_liveness_detection.ipynb         # Liveness detection training
│   ├── 04_emotion_detection.ipynb          # Emotion detection training
│   ├── GPU_test.ipynb                      # GPU testing
│   └── kaggle/                             # Kaggle notebooks
│       ├── emotion-detection.ipynb
│       ├── facial-recognition.ipynb
│       └── liveness-detection.ipynb
│
├── facial_recognition_app/                  # Web application
│   ├── src/
│   │   ├── app.py                          # Main Streamlit app
│   │   ├── components/                     # UI components
│   │   │   ├── face_detector.py
│   │   │   └── emotion_detector.py
│   │   ├── config/                         # Configuration files
│   │   └── utils/                          # Utility functions
│   │       └── face_processor.py
│   ├── data/
│   │   ├── embeddings/                     # Stored face embeddings
│   │   ├── face_db/                        # Face database
│   │   └── faces/                          # Face images
│   ├── models/                             # Model copies for app
│   └── requirements.txt
│
├── facialRecognitionSystem_enhanced.py      # Enhanced standalone system
├── facialRegconitionSystem.py              # Basic standalone system
├── evaluate_system.py                       # Performance evaluation script
├── test_emotion.py                         # Emotion detection testing
├── requirements.txt                         # Project dependencies
└── README.md                               # This file
```

## 🎯 Quick Start

### Option 1: Web Application (Streamlit)

Launch the interactive web interface:

```bash
cd facial_recognition_app
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Option 2: Standalone Python System

Run the enhanced facial recognition system:

```bash
python facialRecognitionSystem_enhanced.py
```

**Interactive Commands:**
- `r` - Register a new face
- `l` - Start liveness challenge
- `q` - Quit the application

## 🤖 Models

### 1. Face Embedding Model
- **Architecture**: Deep CNN with triplet loss
- **Input**: 160x160 RGB face images
- **Output**: 128-dimensional embedding vector
- **Purpose**: Generate unique face representations for recognition and verification

### 2. Emotion Detection Model
- **Architecture**: CNN classifier
- **Input**: 48x48 grayscale face images
- **Output**: 7 emotion classes
- **Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 3. Liveness Detection Model
- **Architecture**: CNN binary classifier
- **Input**: Face images
- **Output**: Real/Fake probability
- **Purpose**: Detect spoofing attacks (photos, videos, masks)

## 💻 Usage

### Registering New Faces

**Standalone System:**
```python
# Press 'r' during runtime
# Follow prompts to enter name
# System will capture and save face embedding
```

**Web Application:**
- Navigate to "Register New Person" tab
- Enter name and upload image
- Click "Register" button

### Face Recognition

The system automatically detects and recognizes faces in real-time:
- **Green box**: Recognized face with name
- **Red box**: Unknown face
- **Yellow box**: Liveness challenge active

### Emotion Detection

Real-time emotion is displayed above each detected face with confidence score.

### Liveness Detection

**Challenge Types:**
1. **Blink Detection**: User must blink their eyes
2. **Smile Detection**: User must smile
3. **Neutral Expression**: User must maintain neutral expression

Press 'l' to start a random challenge in standalone mode.

## 📓 Notebooks

The project includes comprehensive Jupyter notebooks for training and experimentation:

1. **01_data_processing.ipynb**: Data loading, preprocessing, and augmentation
2. **02_face_verification.ipynb**: Training face verification model with triplet loss
3. **02b_compare_models.ipynb**: Compare different model architectures and approaches
4. **03_liveness_detection.ipynb**: Train and evaluate liveness detection model
5. **04_emotion_detection.ipynb**: Train emotion recognition model
6. **GPU_test.ipynb**: Test GPU availability and performance

### Kaggle Notebooks

Pre-trained models and experiments available on Kaggle:
- Emotion detection training
- Facial recognition with different architectures
- Liveness detection experiments

## 📊 Evaluation

Evaluate system performance:

```bash
python evaluate_system.py
```

**Metrics Evaluated:**
- Face Recognition: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Emotion Detection: Per-class accuracy, confusion matrix
- Liveness Detection: True Positive Rate, False Positive Rate

**Output:**
- Performance metrics printed to console
- Confusion matrices saved as images
- ROC curves generated
- Detailed classification reports

## 📦 Requirements

### Core Dependencies

```
numpy==1.24.3
tensorflow==2.15.0
keras==2.15
opencv-python==4.8.1.78
h5py==3.10.0
pandas==2.1.3
scipy==1.11.4
Pillow==10.1.0
```

### Web Application

```
streamlit==1.28.0
streamlit-webrtc==0.47.1
av==10.0.0
mediapipe==0.10.8
```

### Additional Tools

```
matplotlib (for visualization)
seaborn (for confusion matrices)
scikit-learn (for metrics)
```

See `requirements.txt` for complete list.

## 🔧 Configuration

### Face Recognition Thresholds

Edit in `facialRecognitionSystem_enhanced.py`:
```python
threshold = 0.6  # Lower = stricter matching
```

### Emotion Detection Confidence

Minimum confidence threshold for emotion display:
```python
min_confidence = 0.3
```

### Liveness Challenge Settings

```python
challenge_max_frames = 120  # 4 seconds at 30fps
blink_threshold = 2  # Minimum blinks required
```

## 🎨 Web Application Features

The Streamlit web application includes:

- **Live Camera Feed**: Real-time face detection and recognition
- **Face Registration**: Easy interface to register new people
- **Database Management**: View and manage registered faces
- **Emotion Analytics**: Track emotion statistics over time
- **Attendance System**: Log recognized faces with timestamps
- **Modern UI**: Responsive design with gradient cards and smooth animations

## 🧪 Testing

Test individual components:

```bash
# Test emotion detection
python test_emotion.py

# Test full system
python evaluate_system.py
```

## 📈 Performance

**Typical Performance Metrics:**
- Face Recognition Accuracy: ~95%+
- Emotion Detection Accuracy: ~85-90%
- Liveness Detection Accuracy: ~90-95%
- Real-time FPS: 15-30 (depending on hardware)

## 🛠️ Development

### Adding New Features

1. **New Emotion Categories**: Retrain `emotion_detector.keras` with additional classes
2. **Improved Face Recognition**: Fine-tune `embedding_model.keras` on domain-specific data
3. **Enhanced Liveness**: Add more challenge types in `facialRecognitionSystem_enhanced.py`

### Custom Model Training

Use the provided notebooks to train models on your own datasets:
1. Prepare data in the format specified in `01_data_processing.ipynb`
2. Follow training procedures in respective notebooks
3. Replace models in `models/` directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **blacki0214** - [GitHub Profile](https://github.com/blacki0214)

## 🙏 Acknowledgments

- TensorFlow and Keras teams for deep learning frameworks
- OpenCV community for computer vision tools
- Streamlit for the web framework
- Kaggle for datasets and computational resources

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system is designed for educational and research purposes. For production deployments, additional security measures and performance optimizations should be implemented.
