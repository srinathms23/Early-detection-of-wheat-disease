"""
Training Script — Wheat Disease Detection (scikit-learn)
Usage: python train.py --create-dataset --epochs 30
"""

import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import WheatDiseaseDetector
from data_processing import DataPreprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--create-dataset',    action='store_true')
    p.add_argument('--dataset-path',      default='./wheat_dataset')
    p.add_argument('--model-path',        default='./models')
    p.add_argument('--model-type',        default='rf')
    p.add_argument('--epochs',            type=int, default=30)   # kept for compat
    p.add_argument('--batch-size',        type=int, default=16)   # kept for compat
    p.add_argument('--samples-per-class', type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*60)
    print("  WHEAT DISEASE DETECTION — TRAINING")
    print("="*60)

    preprocessor = DataPreprocessor()

    if args.create_dataset:
        preprocessor.create_sample_dataset(
            args.dataset_path, args.samples_per_class
        )

    print(f"Loading dataset: {args.dataset_path}")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.load_and_preprocess(args.dataset_path)

    detector = WheatDiseaseDetector()
    detector.build_model()

    print("Training Random Forest...")
    history = detector.train(
        X_train, y_train,
        val_paths=X_val, val_labels=y_val
    )

    print("Evaluating on test set...")
    metrics = detector.evaluate(X_test, y_test)

    # Save model
    os.makedirs(args.model_path, exist_ok=True)
    model_file = os.path.join(args.model_path, 'wheat_model.pkl')
    detector.save_model(model_file)

    # Save summary
    os.makedirs('./output', exist_ok=True)
    summary = {
        'model_type': 'RandomForest',
        'test_accuracy': round(float(metrics['accuracy']), 4),
        'train_accuracy': round(float(history['accuracy'][0]), 4),
        'val_accuracy': round(float(history['val_accuracy'][0]), 4) if history['val_accuracy'] else None,
        'samples_per_class': args.samples_per_class,
    }
    with open('./output/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Simple accuracy bar chart
    labels = ['Train', 'Validation', 'Test']
    values = [
        summary['train_accuracy'] * 100,
        (summary['val_accuracy'] or 0) * 100,
        summary['test_accuracy'] * 100
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Wheat Disease Detection — Model Accuracy')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output/training_history.png', dpi=150)
    plt.close()

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  Test Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Model saved    : {model_file}")
    print(f"  Summary        : ./output/training_summary.json")
    print("="*60)
    print("\n  Now run:  python app.py")
    print("  Open:     http://localhost:5000\n")


if __name__ == '__main__':
    main()
