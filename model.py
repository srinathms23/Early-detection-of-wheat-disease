"""
Wheat Disease Detection Model — scikit-learn
Uses Random Forest on color/texture features extracted from images.
No DLL dependencies, runs on any Python 3.x installation.
"""

import os
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ['Healthy', 'Leaf Rust', 'Powdery Mildew', 'Septoria Leaf Blotch']

DISEASE_INFO = {
    'Healthy': {
        'severity': 'NONE',
        'action': 'No treatment needed. Continue regular monitoring.',
        'chemicals': [],
        'prevention': ['Regular monitoring', 'Maintain soil health', 'Proper irrigation']
    },
    'Leaf Rust': {
        'severity': 'HIGH',
        'action': 'Apply fungicide immediately to prevent spread.',
        'chemicals': ['Mancozeb', 'Propiconazole', 'Azoxystrobin'],
        'prevention': ['Use resistant varieties', 'Monitor weather patterns', 'Apply preventive fungicides']
    },
    'Powdery Mildew': {
        'severity': 'MEDIUM',
        'action': 'Apply sulfur-based fungicide. Improve air circulation.',
        'chemicals': ['Sulfur', 'Potassium bicarbonate', 'Trichoderma'],
        'prevention': ['Maintain plant spacing', 'Avoid high nitrogen', 'Monitor humidity']
    },
    'Septoria Leaf Blotch': {
        'severity': 'HIGH',
        'action': 'Apply protective fungicide. Remove infected residues.',
        'chemicals': ['Mancozeb', 'Chlorothalonil', 'Azoxystrobin'],
        'prevention': ['Crop rotation', 'Remove crop residues', 'Use certified seeds']
    }
}


def extract_features(image_path, size=64):
    """
    Extract color histogram + texture features from an image.
    Returns a 1-D numpy feature vector.
    """
    img = Image.open(image_path).convert('RGB').resize((size, size))
    arr = np.array(img, dtype=np.float32)

    features = []

    # ── Per-channel statistics (mean, std, min, max) ──────────────────────────
    for ch in range(3):
        ch_data = arr[:, :, ch].flatten()
        features += [
            float(np.mean(ch_data)),
            float(np.std(ch_data)),
            float(np.min(ch_data)),
            float(np.max(ch_data)),
            float(np.percentile(ch_data, 25)),
            float(np.percentile(ch_data, 75)),
        ]

    # ── Color histograms (16 bins per channel) ────────────────────────────────
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch].flatten(), bins=16, range=(0, 256))
        features += (hist / (size * size)).tolist()   # normalise

    # ── Grayscale texture (gradient magnitude statistics) ─────────────────────
    gray = arr.mean(axis=2)

    # Horizontal / vertical gradients
    gx = np.abs(np.diff(gray, axis=1)).flatten()
    gy = np.abs(np.diff(gray, axis=0)).flatten()
    for g in (gx, gy):
        features += [float(np.mean(g)), float(np.std(g)), float(np.max(g))]

    # ── Green-ratio feature (key for plant health) ────────────────────────────
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    denom = r + g + b + 1e-6
    green_ratio = (g / denom).flatten()
    features += [float(np.mean(green_ratio)), float(np.std(green_ratio))]

    # ── Orange/brown ratio (rust indicator) ───────────────────────────────────
    orange_mask = ((r > 150) & (g < 120) & (b < 80)).flatten()
    features.append(float(orange_mask.mean()))

    # ── White ratio (mildew indicator) ────────────────────────────────────────
    white_mask = ((r > 200) & (g > 200) & (b > 200)).flatten()
    features.append(float(white_mask.mean()))

    # ── Dark-spot ratio (septoria indicator) ──────────────────────────────────
    dark_mask = ((r < 100) & (g < 100) & (b < 80)).flatten()
    features.append(float(dark_mask.mean()))

    return np.array(features, dtype=np.float32)


class WheatDiseaseDetector:
    """Random Forest classifier for wheat disease detection."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = CLASS_NAMES
        self.is_trained = False

    def build_model(self, model_type='rf'):
        """Build the classifier (model_type ignored, kept for API compatibility)."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        print("Model built: RandomForestClassifier (200 trees)")
        return self.model

    def train(self, image_paths, labels, val_paths=None, val_labels=None,
              epochs=None, model_path='./models'):
        """
        Train on a list of image paths + integer labels.
        Returns a history-like dict for compatibility.
        """
        if self.model is None:
            self.build_model()

        print(f"Extracting features from {len(image_paths)} images...")
        X = np.array([extract_features(p) for p in image_paths])
        y = np.array(labels)

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.is_trained = True

        train_acc = self.model.score(X, y)
        print(f"Train accuracy: {train_acc*100:.2f}%")

        history = {'accuracy': [train_acc], 'loss': [1 - train_acc],
                   'val_accuracy': [], 'val_loss': []}

        if val_paths and val_labels:
            Xv = np.array([extract_features(p) for p in val_paths])
            Xv = self.scaler.transform(Xv)
            val_acc = self.model.score(Xv, np.array(val_labels))
            history['val_accuracy'].append(val_acc)
            history['val_loss'].append(1 - val_acc)
            print(f"Val  accuracy: {val_acc*100:.2f}%")

        return history

    def evaluate(self, image_paths, labels):
        """Evaluate on test set. Returns metrics dict."""
        X = np.array([extract_features(p) for p in image_paths])
        X = self.scaler.transform(X)
        acc = self.model.score(X, np.array(labels))
        print(f"Test Accuracy: {acc*100:.2f}%")
        return {'accuracy': acc}

    def predict(self, image_path):
        """
        Predict disease for a single image path.
        Returns dict compatible with Flask API and HTML frontend.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Run train.py first.")

        feat = extract_features(image_path).reshape(1, -1)
        feat = self.scaler.transform(feat)

        probs = self.model.predict_proba(feat)[0]
        idx = int(np.argmax(probs))
        disease = self.class_names[idx]
        confidence = float(probs[idx]) * 100

        all_preds = {
            self.class_names[i]: f"{float(probs[i])*100:.2f}%"
            for i in range(len(self.class_names))
        }

        info = DISEASE_INFO.get(disease, {})
        return {
            'disease': disease,
            'confidence': f"{confidence:.2f}%",
            'confidence_raw': confidence,
            'all_predictions': all_preds,
            'severity': info.get('severity', 'UNKNOWN'),
            'action': info.get('action', ''),
            'chemicals': info.get('chemicals', []),
            'prevention': info.get('prevention', [])
        }

    def save_model(self, path):
        """Save model + scaler to a .pkl file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'class_names': self.class_names}, f)
        print(f"Model saved: {path}")

    def load_model(self, path, **kwargs):
        """Load model + scaler from a .pkl file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.class_names = data.get('class_names', CLASS_NAMES)
        self.is_trained = True
        print(f"Model loaded: {path}")
        return self.model
