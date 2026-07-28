# 🌾 Wheat Disease Detection System

An AI-powered early detection system for wheat diseases using deep learning and computer vision.

## Features

✅ **Automatic Disease Classification**
- Healthy wheat leaves
- Leaf Rust
- Powdery Mildew
- Septoria Leaf Blotch

✅ **Multiple Detection Models**
- MobileNetV2 (Fast, lightweight)
- Custom CNN (For research)
- Transfer learning support

✅ **Web Interface**
- Drag-and-drop image upload
- Real-time predictions
- Detailed disease information
- Management recommendations

✅ **Command-line Tools**
- Batch prediction
- Model training
- Results export (JSON)

✅ **Disease Management Info**
- Severity levels
- Recommended actions
- Chemical recommendations
- Prevention strategies

---

## Project Structure

```
wheat_disease_detection/
├── model.py                 # CNN model class
├── data_processing.py       # Data loading & preprocessing
├── train.py                # Training script
├── test.py                 # Testing & inference script
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── models/                 # Trained models (auto-created)
├── wheat_dataset/          # Dataset directory (auto-created)
├── uploads/               # Uploaded images (auto-created)
└── output/                # Results & logs (auto-created)
```

---

## Installation

### 1. Clone or Download Project

```bash
cd wheat_disease_detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may require additional setup on some systems. For GPU support:

```bash
pip install tensorflow[and-cuda]  # For NVIDIA GPU
```

### 3. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## Quick Start

### Option 1: Train from Scratch (Recommended for First Run)

```bash
# Create synthetic dataset and train model
python train.py --create-dataset --epochs 50 --model-type mobilenet
```

### Option 2: Launch Web Interface

```bash
# Start Flask web app
python app.py
```

Then open browser to: **http://localhost:5000**

### Option 3: Test Single Image

```bash
# Test with single image
python test.py --image path/to/leaf.jpg --visualize --save
```

### Option 4: Batch Processing

```bash
# Process multiple images
python test.py --batch-folder ./wheat_dataset/Healthy --model-path ./models/wheat_disease_model_mobilenet.h5
```

---

## Detailed Usage

### Training from Scratch

**Step 1:** Create synthetic dataset

```bash
python train.py \
  --create-dataset \
  --samples-per-class 50 \
  --dataset-path ./wheat_dataset \
  --model-path ./models \
  --model-type mobilenet \
  --epochs 50 \
  --batch-size 32 \
  --visualize
```

**Parameters:**
- `--create-dataset`: Generate synthetic data
- `--samples-per-class`: Images per disease class
- `--model-type`: Choose `mobilenet` (fast) or `custom` (lightweight)
- `--epochs`: Training rounds
- `--batch-size`: Images per batch
- `--visualize`: Show data samples

**Output:**
- ✓ Trained model saved to `./models/`
- ✓ Training history plot
- ✓ Performance summary (JSON)

### Single Image Prediction

```bash
python test.py \
  --image ./uploads/wheat_leaf.jpg \
  --model-path ./models/wheat_disease_model_mobilenet.h5 \
  --visualize \
  --save
```

**Output:**
- Disease classification
- Confidence score
- All prediction probabilities
- Management recommendations
- Saved as JSON

### Batch Prediction

```bash
python test.py \
  --batch-folder ./wheat_dataset/ \
  --model-path ./models/wheat_disease_model_mobilenet.h5
```

**Output:**
- Prediction for each image
- Summary statistics
- Saved as JSON

### Web Application

```bash
python app.py
```

**Features:**
- Upload images via drag-and-drop
- Real-time predictions
- Disease information
- Prediction history
- Statistics dashboard

Access: **http://localhost:5000**

---

## Disease Reference

### 1. Healthy 🟢

- **Status:** No disease detected
- **Action:** Continue monitoring
- **Treatment:** None required

### 2. Leaf Rust 🔴

- **Symptoms:** Orange/reddish pustules on leaves
- **Severity:** HIGH
- **Action:** Apply fungicide immediately
- **Chemicals:** Mancozeb, Propiconazole, Azoxystrobin
- **Prevention:**
  - Use resistant varieties
  - Monitor weather patterns
  - Apply preventive fungicides

### 3. Powdery Mildew 🟡

- **Symptoms:** White powdery coating on leaves
- **Severity:** MEDIUM
- **Action:** Apply sulfur-based fungicide
- **Chemicals:** Sulfur, Potassium bicarbonate, Trichoderma
- **Prevention:**
  - Maintain plant spacing
  - Avoid high nitrogen
  - Monitor humidity

### 4. Septoria Leaf Blotch 🔴

- **Symptoms:** Gray/brown rectangular spots with dark borders
- **Severity:** HIGH
- **Action:** Apply protective fungicide
- **Chemicals:** Mancozeb, Chlorothalonil, Azoxystrobin
- **Prevention:**
  - Crop rotation
  - Remove residues
  - Use certified seeds

---

## Model Architecture

### MobileNetV2 (Recommended)
- **Speed:** ⚡⚡⚡ Very Fast
- **Accuracy:** ⭐⭐⭐⭐ High
- **Size:** 36 MB
- **Use Case:** Production deployment
- **Transfer Learning:** ImageNet pretrained

### Custom CNN
- **Speed:** ⚡⚡⚡⭐ Fast
- **Accuracy:** ⭐⭐⭐ Good
- **Size:** 5 MB
- **Use Case:** Research, edge deployment
- **Architecture:** 3 CNN blocks + Dense layers

---

## Performance Metrics

After training, check results in `output/training_summary.json`:

```json
{
  "model_type": "mobilenet",
  "image_size": 224,
  "epochs_trained": 45,
  "batch_size": 32,
  "test_results": {
    "accuracy": 0.9234,
    "precision": 0.9156,
    "recall": 0.9301,
    "loss": 0.2145
  }
}
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution:**
```bash
python train.py --create-dataset --epochs 30
```

### Issue: Out of Memory (OOM) Error

**Solution:**
```bash
python train.py --batch-size 16 --epochs 30
```

### Issue: Slow Training

**Solution:**
- Use MobileNetV2: `--model-type mobilenet`
- Reduce epochs: `--epochs 20`
- Use GPU: Install `tensorflow[and-cuda]`

### Issue: Web app not opening

**Solution:**
```bash
# Check if port 5000 is available
# Or use different port:
python app.py --port 8000
```

---

## Dataset Information

### Synthetic Dataset Created

The script automatically creates a synthetic dataset with:
- **4 Classes:** Healthy, Leaf Rust, Powdery Mildew, Septoria
- **Images per class:** 20-50 (configurable)
- **Image size:** 224x224 pixels
- **Format:** JPEG

### Using Real Dataset

To use real wheat leaf images:

```
wheat_dataset/
├── Healthy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Leaf Rust/
│   ├── img1.jpg
│   └── ...
├── Powdery Mildew/
│   └── ...
└── Septoria Leaf Blotch/
    └── ...
```

Then run:
```bash
python train.py --dataset-path ./wheat_dataset --epochs 50
```

---

## API Reference

### Model Class

```python
from model import WheatDiseaseDetector

# Initialize
detector = WheatDiseaseDetector(num_classes=4, img_size=224)

# Build model
detector.build_model(model_type='mobilenet')

# Train
history = detector.train(train_data, val_data, epochs=50)

# Predict
result = detector.predict('leaf.jpg')
# Output:
# {
#   'disease': 'Leaf Rust',
#   'confidence': '92.34%',
#   'all_predictions': {...}
# }

# Save
detector.save_model('model.h5')

# Load
detector.load_model('model.h5')
```

### Data Processing Class

```python
from data_processing import DataPreprocessor

preprocessor = DataPreprocessor(img_size=224)

# Create synthetic data
preprocessor.create_sample_dataset('./wheat_dataset', num_samples_per_class=30)

# Load and split data
train_data, val_data, test_data = preprocessor.load_and_preprocess('./wheat_dataset')

# Visualize
preprocessor.visualize_dataset('./wheat_dataset')

# Get distribution
distribution = preprocessor.get_class_distribution('./wheat_dataset')
```

---

## Results & Output Files

After training/testing, check these files:

### Training Output
- `training_history.png` - Loss/Accuracy graphs
- `training_summary.json` - Model performance metrics
- `models/wheat_disease_model_mobilenet.h5` - Trained model

### Inference Output
- `prediction.json` - Single image results
- `prediction_result.png` - Visualization
- `batch_prediction_results.json` - Batch results

---

## Performance Optimization

### Speed Up Training
```bash
python train.py --model-type mobilenet --batch-size 64 --epochs 30
```

### Improve Accuracy
```bash
python train.py --model-type mobilenet --batch-size 16 --epochs 100
```

### Reduce Model Size
```bash
# Use custom CNN (5MB vs 36MB)
python train.py --model-type custom
```

---

## Future Enhancements

- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Real-time camera detection
- [ ] Edge device support (Raspberry Pi)
- [ ] Ensemble models for better accuracy
- [ ] Attention mechanisms for interpretability
- [ ] Multi-crop analysis
- [ ] Disease severity grading
- [ ] Treatment effectiveness tracking

---

## Dataset Sources

For real wheat leaf images, consider:
- [Kaggle Datasets](https://www.kaggle.com/search?q=wheat+disease)
- [PlantVillage Dataset](https://plantvillage.psu.edu/)
- [UC Davis Wheat Dataset](https://plantbreak.org/)

---

## Citations & References

```
@dataset{wheat_disease_2024,
  title={Wheat Disease Detection Using Deep Learning},
  year={2024},
  url={https://github.com/yourusername/wheat-disease-detection}
}
```

---

## License

This project is open-source and available under the MIT License.

---

## Support & Contributing

For issues, questions, or contributions:

1. Check existing issues
2. Create detailed bug reports
3. Submit pull requests
4. Share improvements

---

## Contact & Credits

**Developed with:**
- TensorFlow & Keras
- OpenCV
- Flask
- Deep Learning

**Version:** 1.0.0
**Last Updated:** 2024

---

## Disclaimer

This system is for educational and research purposes. For commercial agricultural use, consult with agricultural experts and conduct proper validation with your specific wheat varieties and growing conditions.

---

**Happy Disease Detecting! 🌾🤖**
