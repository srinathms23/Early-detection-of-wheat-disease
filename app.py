"""
Flask Web Application — Wheat Disease Detection
Run: python app.py
Open: http://localhost:5000
"""

import os
import uuid
import argparse
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

os.makedirs('./uploads', exist_ok=True)
os.makedirs('./models',  exist_ok=True)

detector = None

# Load model at import time so gunicorn workers have it ready
load_detector()


def load_detector():
    global detector
    try:
        from model import WheatDiseaseDetector
        detector = WheatDiseaseDetector()

        candidates = [
            './models/wheat_model.pkl',
            './models/wheat_model_rf.pkl',
        ]
        for path in candidates:
            if os.path.exists(path):
                detector.load_model(path)
                print(f"✓ Model loaded: {path}")
                return True

        print("⚠  No trained model found.")
        print("   Run first:  python train.py --create-dataset")
        return False
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use JPG or PNG.'}), 400

    if detector is None or not detector.is_trained:
        return jsonify({
            'error': 'Model not ready. Run: python train.py --create-dataset'
        }), 503

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Validate image
    try:
        img = Image.open(filepath)
        img.verify()
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'Invalid or corrupted image.'}), 400

    try:
        result = detector.predict(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/health')
def health():
    ready = detector is not None and detector.is_trained
    return jsonify({
        'status': 'ok',
        'model_loaded': ready,
        'message': 'Ready' if ready else 'Model not loaded — run train.py first'
    })


@app.route('/classes')
def classes():
    from model import CLASS_NAMES, DISEASE_INFO
    return jsonify({'classes': CLASS_NAMES, 'info': DISEASE_INFO})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',  type=int, default=int(os.environ.get('PORT', 5000)))
    parser.add_argument('--host',  default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  WHEAT DISEASE DETECTION — WEB APP")
    print("="*55)

    load_detector()

    print(f"\n  Open browser: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
