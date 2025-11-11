# Quick Start Guide - Facial Recognition System

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

Open terminal/command prompt and run:

```bash
cd d:\apply_ML\facial_recognition
pip install -r requirements.txt
```

### Step 2: Verify Models

Make sure these files exist in `models/` folder:
- ✅ `emotion_detector.keras`
- ✅ `embedding_model.keras` 
- ✅ `liveness_detector_zalo.keras` (optional)

### Step 3: Run the System

```bash
python facialRecognitionSystem_enhanced.py
```

## 📋 First Time Usage

### 1. Register Your Face

1. When the application starts, press **`r`**
2. Type your name and press Enter
3. Look at the camera
4. Your face will be registered automatically
5. You'll see a confirmation message

### 2. Test Recognition

1. Press **`d`** for detection mode
2. Look at the camera
3. You should see:
   - Your name with confidence score
   - Current emotion
   - Liveness status (if model available)

### 3. Try Liveness Challenge

1. Press **`l`** for liveness mode
2. Follow the on-screen instructions:
   - "BLINK YOUR EYES TWICE" - Blink naturally
   - "SMILE FOR 2 SECONDS" - Smile and hold
   - "KEEP A NEUTRAL FACE" - Stay neutral
3. Challenge passes when complete ✓

## ⌨️ Complete Controls Reference

| Key | Action | Description |
|-----|--------|-------------|
| `q` | Quit | Exit the application |
| `r` | Register | Add new face to database |
| `d` | Detection | Normal recognition mode (default) |
| `l` | Liveness | Start liveness challenge |
| `s` | Show | List all registered faces |
| `x` | Delete | Remove a registered face |

## 🎯 Common Tasks

### Register Multiple People

```
1. Press 'r'
2. Enter "Alice"
3. Press 'r' again
4. Enter "Bob"
5. Repeat for each person
```

### Check Who's Registered

```
Press 's' at any time to see list of registered faces
```

### Delete a Face

```
1. Press 'x'
2. Type the name to delete
3. Confirm
```

### Test Emotion Detection

Try making different facial expressions:
- 😠 Angry face
- 😃 Happy/Smile
- 😮 Surprised
- 😐 Neutral
- 😢 Sad
- 😨 Fear
- 🤢 Disgust

The system will detect and display your emotion in real-time!

## 🔧 Troubleshooting

### Problem: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow
```

### Problem: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
```

### Problem: Webcam not detected

**Solution:**
- Check if webcam is connected
- Close other apps using webcam (Zoom, Skype, etc.)
- Try changing camera index in code (line with `cv2.VideoCapture(0)`)

### Problem: Models not loading

**Solution:**
- Verify models exist in `models/` folder
- Check file names match exactly
- Re-download models if corrupted

### Problem: Poor recognition accuracy

**Solution:**
- Ensure good lighting
- Face the camera directly
- Register face from multiple angles
- Remove glasses/hats if possible
- Clean camera lens

### Problem: Liveness challenges timeout

**Solution:**
- Perform actions more clearly
- Ensure face is fully visible
- Move closer to camera
- Improve lighting

## 📊 Running Performance Evaluation

To generate comprehensive performance metrics:

```bash
python evaluate_system.py
```

This will:
- Evaluate face recognition accuracy
- Test emotion detection performance  
- Measure liveness detection metrics
- Generate visualization plots
- Create detailed report

Results saved in: `output/evaluation/`

## 🎥 Demo Workflow

### Complete Demo Sequence:

1. **Start System**
   ```bash
   python facialRecognitionSystem_enhanced.py
   ```

2. **Register Person 1**
   - Press `r`
   - Enter "Alice"
   - Face camera
   - Wait for confirmation

3. **Register Person 2**
   - Press `r`
   - Enter "Bob"
   - Have Bob face camera
   - Wait for confirmation

4. **Show Registered Faces**
   - Press `s`
   - See list of Alice and Bob

5. **Test Recognition**
   - Press `d`
   - Alice faces camera → System shows "Alice"
   - Bob faces camera → System shows "Bob"
   - Unknown person → System shows "Unknown"

6. **Test Emotion Detection**
   - Smile → "Happy"
   - Frown → "Sad" or "Angry"
   - Surprise face → "Surprise"
   - Neutral → "Neutral"

7. **Test Liveness**
   - Press `l`
   - Follow blink challenge
   - Challenge passes ✓
   - Back to detection mode

8. **Quit**
   - Press `q`

## 💾 Data Storage

### Where is data saved?

- **Registered faces**: `data/face_database.pkl`
- **Evaluation results**: `output/evaluation/`
- **Sample images**: `output/`

### Backup your database

```bash
# Windows
copy data\face_database.pkl data\face_database_backup.pkl

# Linux/Mac
cp data/face_database.pkl data/face_database_backup.pkl
```

### Reset database (start fresh)

```bash
# Windows
del data\face_database.pkl

# Linux/Mac
rm data/face_database.pkl
```

## 📝 Tips for Best Results

### For Face Registration:
✅ Good lighting (front-facing light)
✅ Neutral expression
✅ Face camera directly
✅ Remove glasses temporarily
✅ Clear background
✅ Normal distance (arm's length)

### For Emotion Detection:
✅ Make exaggerated expressions
✅ Hold expression for 1-2 seconds
✅ Ensure face is fully visible
✅ Good lighting on face

### For Liveness Challenges:
✅ Perform actions clearly
✅ Don't rush
✅ Face camera directly
✅ Make sure eyes are visible for blink detection

## 🐛 Known Limitations

1. **Lighting**: Poor lighting affects accuracy
2. **Angle**: Works best with frontal face view
3. **Occlusion**: Masks/sunglasses reduce accuracy
4. **Multiple Faces**: Processes all detected faces
5. **Distance**: Optimal range is 0.5-2 meters

## 📚 Next Steps

1. ✅ Complete this quick start
2. ✅ Register your face
3. ✅ Test all features
4. ✅ Try liveness challenges
5. ✅ Run evaluation script
6. ✅ Review performance metrics
7. ✅ Read full README.md
8. ✅ Check notebooks for training details

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Review `PROJECT_ASSESSMENT.md` for requirements analysis
3. Examine Jupyter notebooks in `notebooks/` folder
4. Open GitHub issue for bugs
5. Check error messages in terminal

## 🎉 Success Checklist

- [ ] System runs without errors
- [ ] Webcam displays video
- [ ] Face detection works (green boxes)
- [ ] Registration successful
- [ ] Recognition shows correct name
- [ ] Emotion changes with expressions
- [ ] Liveness challenge completable
- [ ] All controls work (q, r, d, l, s, x)

If all checked ✅, your system is working perfectly!

---

**Ready to start? Run:**

```bash
python facialRecognitionSystem_enhanced.py
```

**Enjoy your Facial Recognition System! 🎯**
