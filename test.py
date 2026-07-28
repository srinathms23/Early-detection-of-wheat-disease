"""
Testing / Inference Script for Wheat Disease Detection
Usage:
  python test.py --image leaf.jpg --visualize
  python test.py --batch-folder ./wheat_dataset/
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from model import WheatDiseaseDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Wheat Disease Inference')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--batch-folder', help='Folder of images for batch processing')
    parser.add_argument('--model-path', default='./models/wheat_disease_model_mobilenet.h5',
                        help='Path to trained model')
    parser.add_argument('--visualize', action='store_true', help='Show/save prediction visualization')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    return parser.parse_args()


def find_model(model_path):
    """Find the best available model file."""
    if os.path.exists(model_path):
        return model_path
    # Try alternatives
    candidates = [
        './models/wheat_disease_model_mobilenet.h5',
        './models/wheat_disease_model_custom.h5',
        './models/best_model.h5',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def visualize_prediction(image_path, result, output_path=None):
    """Save a visualization of the prediction."""
    img = Image.open(image_path).convert('RGB')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: image
    axes[0].imshow(img)
    axes[0].set_title(f"Input: {os.path.basename(image_path)}", fontsize=12)
    axes[0].axis('off')

    # Right: bar chart of predictions
    classes = list(result['all_predictions'].keys())
    confidences = [float(v.replace('%', '')) for v in result['all_predictions'].values()]
    colors = ['#2ecc71' if c == result['disease'] else '#bdc3c7' for c in classes]

    bars = axes[1].barh(classes, confidences, color=colors, edgecolor='white')
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1].set_title(
        f"Prediction: {result['disease']}\nConfidence: {result['confidence']}",
        fontsize=12, fontweight='bold'
    )
    for bar, conf in zip(bars, confidences):
        axes[1].text(conf + 1, bar.get_y() + bar.get_height() / 2,
                     f'{conf:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")
    plt.close()


def predict_single(args):
    """Run prediction on a single image."""
    model_path = find_model(args.model_path)
    if not model_path:
        print("ERROR: No trained model found. Run: python train.py --create-dataset")
        sys.exit(1)

    detector = WheatDiseaseDetector()
    detector.load_model(model_path)

    print(f"\nAnalyzing: {args.image}")
    result = detector.predict(args.image)

    print("\n" + "="*50)
    print(f"  Disease     : {result['disease']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Severity    : {result['severity']}")
    print(f"  Action      : {result['action']}")
    if result['chemicals']:
        print(f"  Chemicals   : {', '.join(result['chemicals'])}")
    print("\n  All Predictions:")
    for cls, conf in result['all_predictions'].items():
        marker = " ◀" if cls == result['disease'] else ""
        print(f"    {cls:<25} {conf}{marker}")
    print("="*50)

    if args.visualize:
        os.makedirs('./output', exist_ok=True)
        vis_path = f"./output/prediction_result.png"
        visualize_prediction(args.image, result, vis_path)

    if args.save:
        os.makedirs('./output', exist_ok=True)
        out_path = './output/prediction.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved: {out_path}")

    return result


def predict_batch(args):
    """Run prediction on all images in a folder."""
    model_path = find_model(args.model_path)
    if not model_path:
        print("ERROR: No trained model found. Run: python train.py --create-dataset")
        sys.exit(1)

    detector = WheatDiseaseDetector()
    detector.load_model(model_path)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.batch_folder, '**', ext), recursive=True))

    if not image_files:
        print(f"No images found in: {args.batch_folder}")
        sys.exit(1)

    print(f"\nBatch processing {len(image_files)} images...\n")
    results = []
    disease_counts = {}

    for i, img_path in enumerate(image_files, 1):
        try:
            result = detector.predict(img_path)
            result['image'] = os.path.basename(img_path)
            results.append(result)
            disease_counts[result['disease']] = disease_counts.get(result['disease'], 0) + 1
            print(f"  [{i:03d}/{len(image_files)}] {os.path.basename(img_path):<40} "
                  f"→ {result['disease']} ({result['confidence']})")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    # Summary
    print("\n" + "="*50)
    print("  BATCH SUMMARY")
    print("="*50)
    for disease, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {disease:<30} {count:>3} images ({pct:.1f}%)")
    print("="*50)

    if args.save:
        os.makedirs('./output', exist_ok=True)
        out_path = './output/batch_prediction_results.json'
        with open(out_path, 'w') as f:
            json.dump({'summary': disease_counts, 'results': results}, f, indent=2)
        print(f"\nBatch results saved: {out_path}")

    return results


def main():
    args = parse_args()

    if not args.image and not args.batch_folder:
        print("Provide --image <path> or --batch-folder <path>")
        print("Example: python test.py --image leaf.jpg --visualize")
        sys.exit(1)

    if args.image:
        predict_single(args)
    elif args.batch_folder:
        predict_batch(args)


if __name__ == '__main__':
    main()
