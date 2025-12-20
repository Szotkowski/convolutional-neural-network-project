import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "all_12000_images/"  # Your big folder
TARGET_DIR = "selected_2000_images/"
MAX_IMAGES = 2000

def get_image_quality(image_path):
    """Calculates image sharpness using the Laplacian variance."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return -1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Laplacian variance is a classic measure for sharpness
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return -1

def select_best_images():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_scores = []

    print(f"🔍 Analyzing {len(all_files)} images for quality...")
    
    for filename in tqdm(all_files):
        path = os.path.join(SOURCE_DIR, filename)
        score = get_image_quality(path)
        if score > 0:
            image_scores.append((filename, score))

    # Sort by score descending (highest sharpness first)
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # Pick top 2000
    best_images = image_scores[:MAX_IMAGES]

    print(f"📦 Copying top {MAX_IMAGES} images to {TARGET_DIR}...")
    for filename, score in tqdm(best_images):
        shutil.copy(os.path.join(SOURCE_DIR, filename), os.path.join(TARGET_DIR, filename))

    print(f"✨ Done! Best {MAX_IMAGES} images are in {TARGET_DIR}")

if __name__ == "__main__":
    select_best_images()