# Facial Recognition System with Emotion and Liveness Detection

A comprehensive real-time facial recognition system that combines face recognition, emotion detection, and liveness detection to provide secure and robust identity verification.

## 📋 Features

### 1. **Face Recognition**
- Register new faces with names
- Real-time face recognition from webcam
- Cosine similarity-based matching
- Confidence scores for each prediction
- Face database management (add/delete faces)

### 2. **Emotion Detection**
- Detects 7 emotions in real-time:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Displays emotion with confidence score

### 3. **Liveness Detection**
- **Model-based detection**: Uses deep learning to detect spoofing attacks
- **Challenge-response system**: Interactive verification
  - Blink detection (blink twice)
  - Smile detection (smile for 2 seconds)
  - Neutral face detection (keep neutral face)
- Real-time liveness verification

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/blacki0214/Facial-Recognition.git
cd facial_recognition

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Run the enhanced facial recognition system
python facialRecognitionSystem_enhanced.py
```

### Controls

- **`q`** - Quit the application
- **`r`** - Register new face
- **`d`** - Detection mode (default)
- **`l`** - Liveness challenge mode
- **`s`** - Show registered faces
- **`x`** - Delete a registered face

## 📁 Project Structure

```
facial_recognition/
├── models/
│   ├── emotion_detector.keras          # Emotion detection model
│   ├── embedding_model.keras            # Face embedding model
│   ├── softmax_model.keras              # Alternative face model
│   ├── triplet_model.keras              # Triplet loss trained model
│   └── liveness_detector_zalo.keras     # Liveness detection model
├── data/
│   ├── face_database.pkl                # Registered faces database
│   ├── verification_pairs_test.txt      # Test pairs for evaluation
│   ├── verification_pairs_val.txt       # Validation pairs
│   └── classification_data/             # Training/test data
├── notebooks/
│   ├── 01_data_processing.ipynb         # Data preprocessing
│   ├── 02_face_verfication.ipynb        # Face recognition training
│   ├── 02b_compare_models.ipynb         # Model comparison
│   ├── 03_liveness_detection.ipynb      # Liveness model training
│   └── 04_emotion_detection.ipynb       # Emotion model training
├── output/
│   └── evaluation/                      # Evaluation results
├── facialRecognitionSystem_enhanced.py  # Main application
├── evaluate_system.py                   # Performance evaluation
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## 🔧 System Requirements

- Python 3.7+
- Webcam
- At least 4GB RAM
- GPU recommended for faster inference (optional)

## 📊 Performance Metrics

Run the evaluation script to generate performance metrics:

```bash
python evaluate_system.py
```

This generates:
- Accuracy, Precision, Recall, F1-Score for all modules
- Confusion matrices
- ROC curves
- Detailed evaluation report

### Expected Performance

| Module | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Face Recognition | 95%+ | 93%+ | 94%+ | 93%+ |
| Emotion Detection | 85%+ | 84%+ | 83%+ | 83%+ |
| Liveness Detection | 90%+ | 88%+ | 89%+ | 88%+ |

## 🎯 Usage Examples

### 1. Registering a New Face

1. Run the application
2. Press `r` to enter registration mode
3. Enter the person's name when prompted
4. Look at the camera
5. The system will capture and register your face

### 2. Real-time Recognition

1. Press `d` for detection mode (default)
2. The system will:
   - Detect and recognize faces
   - Display names with confidence scores
   - Show detected emotions
   - Indicate liveness status

### 3. Liveness Challenge

1. Press `l` to start liveness challenge
2. Follow the on-screen instructions:
   - Blink twice
   - Smile for 2 seconds
   - Keep a neutral face
3. Challenge passes when completed successfully

## 🛠️ Technical Details

### Models

#### 1. Face Recognition
- **Architecture**: CNN-based embedding model
- **Input**: 160x160 RGB images
- **Output**: 128-dimensional embeddings
- **Training**: Softmax + Triplet loss
- **Similarity**: Cosine distance
- **Threshold**: 0.6 (adjustable)

#### 2. Emotion Detection
- **Architecture**: CNN
- **Input**: 48x48 grayscale images
- **Output**: 7 emotion classes
- **Dataset**: FER-2013 / Custom dataset
- **Accuracy**: ~85%

#### 3. Liveness Detection
- **Architecture**: CNN
- **Input**: 160x160 RGB images
- **Output**: Binary (Real/Fake)
- **Methods**: 
  - Texture analysis
  - Challenge-response
  - Eye blink detection

### Dependencies

```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 🔒 Security Features

1. **Anti-Spoofing**: Liveness detection prevents photo/video attacks
2. **Challenge-Response**: Interactive verification ensures real person
3. **Confidence Thresholds**: Reject low-confidence matches
4. **Multi-Factor**: Combines face recognition + emotion + liveness

## 📈 Model Training

All models were trained on Kaggle. See notebooks for details:

1. **Face Recognition**: `02_face_verfication.ipynb`
2. **Emotion Detection**: `04_emotion_detection.ipynb`
3. **Liveness Detection**: `03_liveness_detection.ipynb`

Training notebooks available in `notebooks/kaggle/`

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution**: Ensure all model files are in the `models/` directory

### Issue: Webcam not working
**Solution**: Check webcam permissions and OpenCV installation

### Issue: Poor recognition accuracy
**Solution**: 
- Ensure good lighting
- Register multiple images per person
- Adjust confidence threshold
- Retrain with more data

### Issue: Liveness model not found
**Solution**: Download the model or the system will use basic liveness checks

## 📝 Future Improvements

- [ ] Add mask detection
- [ ] Multi-face tracking
- [ ] Age and gender detection
- [ ] GUI with Tkinter/PyQt
- [ ] Database integration (SQLite/MongoDB)
- [ ] REST API for remote access
- [ ] Mobile app integration
- [ ] 3D face reconstruction
- [ ] Facial landmark detection

## 📄 License

This project is licensed under the MIT License.

## 👥 Contributors

- **blacki0214** - Initial work

## 🙏 Acknowledgments

- FER-2013 dataset for emotion detection
- VGGFace2 / LFW for face recognition
- Zalo AI Challenge for liveness detection dataset
- Kaggle for computational resources

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact through the repository.

---

**Note**: This system is for educational and research purposes. For production deployment, additional security measures and testing are recommended.
