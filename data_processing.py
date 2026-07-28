"""
Data Processing — Wheat Disease Detection
Synthetic dataset creation + image path loading for scikit-learn pipeline.
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split


CLASS_NAMES = ['Healthy', 'Leaf Rust', 'Powdery Mildew', 'Septoria Leaf Blotch']


class DataPreprocessor:
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.class_names = CLASS_NAMES

    # ── Synthetic dataset ──────────────────────────────────────────────────────

    def create_sample_dataset(self, dataset_path='./wheat_dataset', num_samples_per_class=60):
        """Generate synthetic wheat leaf images for each disease class."""
        print(f"\nCreating synthetic dataset: {dataset_path}")
        os.makedirs(dataset_path, exist_ok=True)

        for cls in self.class_names:
            cls_dir = os.path.join(dataset_path, cls)
            os.makedirs(cls_dir, exist_ok=True)
            print(f"  {cls}: generating {num_samples_per_class} images...")
            for i in range(num_samples_per_class):
                img = self._generate_leaf_image(cls, seed=i)
                fname = f"{cls.replace(' ', '_')}_{i:03d}.jpg"
                img.save(os.path.join(cls_dir, fname), 'JPEG', quality=90)

        total = len(self.class_names) * num_samples_per_class
        print(f"  Done — {total} images across {len(self.class_names)} classes\n")
        return dataset_path

    def _generate_leaf_image(self, disease_class, seed=0):
        """Generate a synthetic leaf image with disease-specific patterns."""
        size = self.img_size
        rng = np.random.default_rng(seed * 17 + abs(hash(disease_class)) % 9999)

        # Base: green leaf
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        base_g = int(rng.integers(70, 130))
        noise = rng.integers(-15, 15, size=(size, size, 3))
        arr[:, :, 0] = np.clip(35 + noise[:, :, 0], 0, 255)
        arr[:, :, 1] = np.clip(base_g + noise[:, :, 1], 0, 255)
        arr[:, :, 2] = np.clip(20 + noise[:, :, 2], 0, 255)

        # Vein
        cx = size // 2
        arr[:, max(0, cx-1):min(size, cx+2), :] = [30, min(base_g + 40, 255), 20]

        if disease_class == 'Leaf Rust':
            # Orange-brown pustule spots
            for _ in range(int(rng.integers(15, 35))):
                x, y = rng.integers(5, size - 5, 2)
                r = int(rng.integers(3, 10))
                col = [int(rng.integers(160, 220)),
                       int(rng.integers(60, 110)),
                       int(rng.integers(5, 30))]
                ys, xs = np.ogrid[-r:r+1, -r:r+1]
                mask = xs*xs + ys*ys <= r*r
                y0, y1 = max(0, y-r), min(size, y+r+1)
                x0, x1 = max(0, x-r), min(size, x+r+1)
                my0 = y0 - (y - r)
                mx0 = x0 - (x - r)
                region = arr[y0:y1, x0:x1]
                m = mask[my0:my0+region.shape[0], mx0:mx0+region.shape[1]]
                region[m] = col

        elif disease_class == 'Powdery Mildew':
            # White powdery patches
            for _ in range(int(rng.integers(4, 9))):
                cx2, cy2 = rng.integers(20, size - 20, 2)
                w = int(rng.integers(15, 40))
                h = int(rng.integers(12, 30))
                y0, y1 = max(0, int(cy2)-h), min(size, int(cy2)+h)
                x0, x1 = max(0, int(cx2)-w), min(size, int(cx2)+w)
                alpha = float(rng.uniform(0.5, 0.9))
                arr[y0:y1, x0:x1] = np.clip(
                    arr[y0:y1, x0:x1] * (1 - alpha) +
                    np.array([232, 236, 230]) * alpha, 0, 255
                ).astype(np.uint8)

        elif disease_class == 'Septoria Leaf Blotch':
            # Brown rectangular spots
            for _ in range(int(rng.integers(5, 14))):
                cx2, cy2 = rng.integers(10, size - 10, 2)
                w = int(rng.integers(6, 20))
                h = int(rng.integers(4, 12))
                col = [int(rng.integers(90, 145)),
                       int(rng.integers(65, 100)),
                       int(rng.integers(15, 45))]
                arr[max(0, int(cy2)-h):min(size, int(cy2)+h),
                    max(0, int(cx2)-w):min(size, int(cx2)+w)] = col

        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img

    # ── Data loading ───────────────────────────────────────────────────────────

    def load_and_preprocess(self, dataset_path='./wheat_dataset',
                             val_split=0.15, test_split=0.10, batch_size=None):
        """
        Returns (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels).
        batch_size is accepted but ignored (kept for API compatibility).
        """
        all_paths, all_labels = [], []

        for idx, cls in enumerate(self.class_names):
            cls_dir = os.path.join(dataset_path, cls)
            if not os.path.exists(cls_dir):
                print(f"  Warning: not found — {cls_dir}")
                continue
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {cls}: {len(files)} images")
            for f in files:
                all_paths.append(os.path.join(cls_dir, f))
                all_labels.append(idx)

        print(f"\nTotal: {len(all_paths)} images")

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            all_paths, all_labels, test_size=test_split,
            random_state=42, stratify=all_labels
        )
        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_ratio,
            random_state=42, stratify=y_tmp
        )

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_class_distribution(self, dataset_path='./wheat_dataset'):
        dist = {}
        for cls in self.class_names:
            d = os.path.join(dataset_path, cls)
            dist[cls] = len([f for f in os.listdir(d)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
                        if os.path.exists(d) else 0
        return dist
