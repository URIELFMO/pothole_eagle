import cv2
import os
from pathlib import Path
from src.config import BASE_DIR

def resize_dataset(dataset_root, target_size=640):
    """
    Resizes all images in train/images, valid/images, and test/images 
    under the given dataset_root to the target size.
    """
    dataset_root = Path(dataset_root)
    subsets = ["train", "valid", "test"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for subset in subsets:
        img_dir = dataset_root / subset / "images"
        
        if not img_dir.exists():
            print(f"Skipping {img_dir} (directory not found)")
            continue

        print(f"Processing: {img_dir}")
        
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"  No images found in {subset}")
            continue

        count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            
            # Calculate new size maintaining aspect ratio
            # Scaling the longest side to target_size
            if w > h:
                new_w = target_size
                new_h = int(h * (target_size / w))
            else:
                new_h = target_size
                new_w = int(w * (target_size / h))

            # Resize image
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Save (overwrite)
            cv2.imwrite(str(img_path), resized)
            count += 1
            
        print(f"  Done! Resized {count} images in {subset}")

if __name__ == "__main__":
    # Target the newdataset folder specifically as requested
    NEW_DATASET_PATH = BASE_DIR / "newdataset"
    
    print(f"Targeting dataset at: {NEW_DATASET_PATH}")
    resize_dataset(NEW_DATASET_PATH, target_size=640)
    print("\nAll datasets resized to 640px.")
