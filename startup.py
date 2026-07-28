"""
Startup script — runs once before the web server starts.
Trains the model if no saved model exists (first boot on Render).
"""

import os
import sys

MODEL_PATH = './models/wheat_model.pkl'

if os.path.exists(MODEL_PATH):
    print(f"[startup] Model already exists: {MODEL_PATH}")
    sys.exit(0)

print("[startup] No model found — training now...")

from data_processing import DataPreprocessor
from model import WheatDiseaseDetector

# Generate synthetic dataset + train
preprocessor = DataPreprocessor()
preprocessor.create_sample_dataset('./wheat_dataset', num_samples_per_class=60)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    preprocessor.load_and_preprocess('./wheat_dataset')

detector = WheatDiseaseDetector()
detector.build_model()
detector.train(X_train, y_train, val_paths=X_val, val_labels=y_val)

os.makedirs('./models', exist_ok=True)
detector.save_model(MODEL_PATH)

print(f"[startup] Model saved: {MODEL_PATH}")
