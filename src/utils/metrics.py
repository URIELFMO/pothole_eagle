import pandas as pd
from ultralytics import YOLO

# 1. Define paths to your 4 trained weights
model_paths = {
    "Model_1": "runs/detect/train/weights/best.pt",
    "Model_2": "runs/detect/train-2/weights/best.pt",
    "Model_3": "runs/detect/train-3/weights/best.pt",
    "Model_4": "runs/detect/train-4/weights/best.pt",
    "Model_5": "runs/detect/train-5/weights/best.pt",
    "Model_6": "runs/detect/train-7/weights/best.pt",
}

# Dictionary to hold final benchmark results
benchmark_results = {}

# 2. Loop through each model and evaluate
for name, path in model_paths.items():
    print(f"\n=== Evaluating {name} ===")

    # Load trained model
    model = YOLO(path)

    # Run validation on the test split
    metrics = model.val(
        data=("data_own.yaml" if name == 'Model_6' else "data.yaml"),  # Path to your dataset config
        split="test",  # Explicitly use the test partition
        imgsz=640,  # Keep resolution constant for fair comparison
        batch=16,  # Batch size for evaluation
        device=0,  # GPU ID (or 'cpu')
        plots=True,  # Saves PR curves, confusion matrix to runs/val/
    )

    # 3. Extract key metrics
    benchmark_results[name] = {
        "mAP50": metrics.box.map50,  # mAP at IoU threshold 0.50
        "mAP50-95": metrics.box.map,  # Mean mAP across IoU 0.50-0.95
        "Precision": metrics.results_dict["metrics/precision(B)"],
        "Recall": metrics.results_dict["metrics/recall(B)"],
        "Inference_Speed (ms)": metrics.speed["inference"],
    }

# 4. Convert results to a clean DataFrame for side-by-side comparison
df_compare = pd.DataFrame(benchmark_results).T
print("\n=== FINAL MODEL COMPARISON ===")
print(df_compare)

# Save to CSV for external plotting or reporting
df_compare.to_csv("yolo_models_comparison.csv")