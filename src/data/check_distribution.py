import os
import yaml
from pathlib import Path
from collections import Counter

# Configuration
base_dir = Path(r'C:\Users\morau\Documents\potholeGDL')
yaml_path = base_dir / 'data.yaml'

def check_distribution():
    # Load class names from yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data.get('names', [])
    num_classes = len(class_names)
    
    splits = ['train', 'valid', 'test']
    stats = {split: Counter() for split in splits}

    print(f"{'Split':<10} | {'Total Files':<12} | {'Class Distribution'}")
    print("-" * 60)

    for split in splits:
        label_dir = base_dir / split / 'labels'
        if not label_dir.exists():
            print(f"{split:<10} | {'MISSING':<12} | -")
            continue

        label_files = list(label_dir.glob('*.txt'))
        total_files = len(label_files)
        
        for lb in label_files:
            with open(lb, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        stats[split][class_id] += 1
        
        # Format distribution string
        dist_str = ", ".join([f"{class_names[i]}: {stats[split][i]}" for i in range(num_classes)])
        print(f"{split:<10} | {total_files:<12} | {dist_str}")

    # Calculate percentages for each class across splits
    print("\nClass-wise split ratios:")
    print(f"{'Class':<10} | {'Train %':<10} | {'Val %':<10} | {'Test %':<10}")
    print("-" * 50)
    
    for i in range(num_classes):
        total_instances = sum(stats[split][i] for split in splits)
        if total_instances == 0:
            print(f"{class_names[i]:<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        ratios = [f"{(stats[split][i] / total_instances * 100):.1f}%" for split in splits]
        print(f"{class_names[i]:<10} | {ratios[0]:<10} | {ratios[1]:<10} | {ratios[2]:<10}")

if __name__ == '__main__':
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
    else:
        check_distribution()
