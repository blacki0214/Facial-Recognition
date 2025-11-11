"""
Performance Evaluation Script for Facial Recognition System
Evaluates: Accuracy, Precision, Recall, F1-Score for all modules
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

class SystemEvaluator:
    def __init__(self):
        self.results = {
            'face_recognition': {},
            'emotion_detection': {},
            'liveness_detection': {}
        }
        
        # Load models
        print("Loading models...")
        self.emotion_model = tf.keras.models.load_model("models/emotion_detector.keras")
        self.face_model = tf.keras.models.load_model("models/embedding_model.keras")
        
        try:
            self.liveness_model = tf.keras.models.load_model("models/liveness_detector_zalo.keras")
        except:
            print("Warning: Liveness model not found")
            self.liveness_model = None
        
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        
    def evaluate_face_recognition(self, test_pairs_file="data/verification_pairs_test.txt"):
        """Evaluate face recognition/verification performance"""
        print("\n" + "="*60)
        print("EVALUATING FACE RECOGNITION")
        print("="*60)
        
        # Load test pairs
        pairs = pd.read_csv(test_pairs_file, sep=" ", header=None, names=["img1", "img2", "label"])
        
        y_true = []
        y_pred = []
        similarities = []
        
        for idx, row in pairs.iterrows():
            img1_path = os.path.join("data", row.img1)
            img2_path = os.path.join("data", row.img2)
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                continue
            
            # Get embeddings
            emb1 = self.get_embedding(img1_path)
            emb2 = self.get_embedding(img2_path)
            
            # Calculate similarity
            similarity = self.cosine_similarity(emb1, emb2)
            similarities.append(similarity)
            y_true.append(row.label)
            
            # Threshold-based prediction
            y_pred.append(1 if similarity > 0.5 else 0)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(pairs)} pairs...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, similarities)
        except:
            auc = 0.0
        
        self.results['face_recognition'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'y_true': y_true,
            'y_pred': y_pred,
            'similarities': similarities
        }
        
        print(f"\n✓ Face Recognition Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        return self.results['face_recognition']
    
    def evaluate_emotion_detection(self, test_data_dir="data/emotion_test"):
        """Evaluate emotion detection performance"""
        print("\n" + "="*60)
        print("EVALUATING EMOTION DETECTION")
        print("="*60)
        
        if not os.path.exists(test_data_dir):
            print("⚠ Test data directory not found. Skipping emotion evaluation.")
            return None
        
        y_true = []
        y_pred = []
        
        # Assuming test data is organized by emotion folders
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(test_data_dir, emotion.lower())
            if not os.path.exists(emotion_dir):
                continue
            
            images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]
            
            for img_name in images:
                img_path = os.path.join(emotion_dir, img_name)
                
                # Predict emotion
                predicted_emotion, conf = self.predict_emotion(img_path)
                predicted_idx = self.emotions.index(predicted_emotion)
                
                y_true.append(emotion_idx)
                y_pred.append(predicted_idx)
            
            print(f"  Processed {len(images)} images for {emotion}")
        
        if len(y_true) == 0:
            print("⚠ No test images found")
            return None
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        self.results['emotion_detection'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'labels': self.emotions
        }
        
        print(f"\n✓ Emotion Detection Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print("\n  Per-Emotion Metrics:")
        report = classification_report(y_true, y_pred, target_names=self.emotions, zero_division=0)
        print(report)
        
        return self.results['emotion_detection']
    
    def evaluate_liveness_detection(self, real_dir="data/liveness_test/real", 
                                   fake_dir="data/liveness_test/fake"):
        """Evaluate liveness detection performance"""
        print("\n" + "="*60)
        print("EVALUATING LIVENESS DETECTION")
        print("="*60)
        
        if self.liveness_model is None:
            print("⚠ Liveness model not loaded. Skipping liveness evaluation.")
            return None
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print("⚠ Test data directories not found. Skipping liveness evaluation.")
            return None
        
        y_true = []
        y_pred = []
        y_scores = []
        
        # Process real images
        real_images = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
        for img_name in real_images:
            img_path = os.path.join(real_dir, img_name)
            is_real, confidence = self.predict_liveness(img_path)
            y_true.append(1)  # Real
            y_pred.append(1 if is_real else 0)
            y_scores.append(confidence)
        
        print(f"  Processed {len(real_images)} real images")
        
        # Process fake images
        fake_images = [f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))]
        for img_name in fake_images:
            img_path = os.path.join(fake_dir, img_name)
            is_real, confidence = self.predict_liveness(img_path)
            y_true.append(0)  # Fake
            y_pred.append(1 if is_real else 0)
            y_scores.append(confidence)
        
        print(f"  Processed {len(fake_images)} fake images")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0
        
        self.results['liveness_detection'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"\n✓ Liveness Detection Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        return self.results['liveness_detection']
    
    def get_embedding(self, img_path):
        """Get face embedding from image"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 160))
        img_arr = np.expand_dims(img / 255.0, 0)
        embedding = self.face_model.predict(img_arr, verbose=0).squeeze()
        return embedding
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def predict_emotion(self, img_path):
        """Predict emotion from image"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img_arr = np.expand_dims(np.expand_dims(img, -1), 0) / 255.0
        
        preds = self.emotion_model.predict(img_arr, verbose=0)[0]
        emotion = self.emotions[np.argmax(preds)]
        confidence = np.max(preds)
        
        return emotion, confidence
    
    def predict_liveness(self, img_path):
        """Predict liveness from image"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 160))
        img_arr = np.expand_dims(img / 255.0, 0)
        
        pred = self.liveness_model.predict(img_arr, verbose=0)[0]
        is_real = pred[1] > 0.5 if len(pred) == 2 else pred[0] > 0.5
        confidence = pred[1] if len(pred) == 2 else pred[0]
        
        return is_real, confidence
    
    def generate_visualizations(self, output_dir="output/evaluation"):
        """Generate visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Confusion Matrix for Emotion Detection
        if 'emotion_detection' in self.results and self.results['emotion_detection']:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(self.results['emotion_detection']['y_true'], 
                                self.results['emotion_detection']['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.emotions, yticklabels=self.emotions, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Emotion Detection Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'emotion_confusion_matrix.png'), dpi=300)
            print(f"  ✓ Saved emotion_confusion_matrix.png")
            plt.close()
        
        # 2. ROC Curve for Face Recognition
        if 'face_recognition' in self.results and self.results['face_recognition']:
            fpr, tpr, _ = roc_curve(self.results['face_recognition']['y_true'],
                                   self.results['face_recognition']['similarities'])
            auc = self.results['face_recognition']['auc']
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Face Recognition ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'face_recognition_roc.png'), dpi=300)
            print(f"  ✓ Saved face_recognition_roc.png")
            plt.close()
        
        # 3. Overall Performance Comparison
        metrics_data = []
        for module, results in self.results.items():
            if results:
                metrics_data.append({
                    'Module': module.replace('_', ' ').title(),
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1_score', 0)
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(df))
            width = 0.2
            
            ax.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', color='skyblue')
            ax.bar(x - 0.5*width, df['Precision'], width, label='Precision', color='lightgreen')
            ax.bar(x + 0.5*width, df['Recall'], width, label='Recall', color='lightcoral')
            ax.bar(x + 1.5*width, df['F1-Score'], width, label='F1-Score', color='gold')
            
            ax.set_xlabel('Module')
            ax.set_ylabel('Score')
            ax.set_title('Overall System Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Module'])
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300)
            print(f"  ✓ Saved overall_performance.png")
            plt.close()
    
    def generate_report(self, output_file="output/evaluation/evaluation_report.txt"):
        """Generate text report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FACIAL RECOGNITION SYSTEM - PERFORMANCE EVALUATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            for module, results in self.results.items():
                if results:
                    f.write(f"\n{module.replace('_', ' ').upper()}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Accuracy:  {results.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {results.get('precision', 0):.4f}\n")
                    f.write(f"Recall:    {results.get('recall', 0):.4f}\n")
                    f.write(f"F1-Score:  {results.get('f1_score', 0):.4f}\n")
                    if 'auc' in results:
                        f.write(f"AUC:       {results.get('auc', 0):.4f}\n")
                    f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"\n✓ Saved evaluation report: {output_file}")

def main():
    evaluator = SystemEvaluator()
    
    # Run evaluations
    evaluator.evaluate_face_recognition()
    evaluator.evaluate_emotion_detection()
    evaluator.evaluate_liveness_detection()
    
    # Generate visualizations and report
    evaluator.generate_visualizations()
    evaluator.generate_report()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nCheck the 'output/evaluation' folder for results.")

if __name__ == "__main__":
    main()
