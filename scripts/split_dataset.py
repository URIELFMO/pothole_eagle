import random
import shutil
from pathlib import Path
from src.config import BASE_DIR

# Configuration
train_dir = BASE_DIR / 'train'
val_dir = BASE_DIR / 'valid'
test_dir = BASE_DIR / 'test'

# Split ratios
val_ratio = 0.1
test_ratio = 0.1

def split_dataset():
    images_src = train_dir / 'images'
    labels_src = train_dir / 'labels'

    if not images_src.exists():
        print(f"Error: {images_src} does not exist.")
        return

    # Get all image files
    images = list(images_src.glob('*.jpg')) + list(images_src.glob('*.png')) + list(images_src.glob('*.jpeg'))
    random.shuffle(images)

    total = len(images)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]
    # Rest stay in train_dir

    splits = {
        'valid': val_images,
        'test': test_images
    }

    for split_name, split_images in splits.items():
        dest_img_dir = BASE_DIR / split_name / 'images'
        dest_lbl_dir = BASE_DIR / split_name / 'labels'

        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Moving {len(split_images)} files to {split_name}...")
        for img_path in split_images:
            # Move image
            shutil.move(str(img_path), str(dest_img_dir / img_path.name))
            
            # Move corresponding label
            label_path = labels_src / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.move(str(label_path), str(dest_lbl_dir / label_path.name))
            else:
                print(f"Warning: Label not found for {img_path.name}")

    # Count final distribution
    final_train_count = len(list(images_src.glob('*.*')))
    print(f"\nSplit complete!")
    print(f"Train: {final_train_count} images remaining")
    print(f"Val:   {val_count} images moved")
    print(f"Test:  {test_count} images moved")

if __name__ == '__main__':
    split_dataset()
