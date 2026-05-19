import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Define paths to the training directories of your 4 models
model_dirs = {
    "Model_1": "runs/detect/train",
    "Model_2": "runs/detect/train-2",
    "Model_3": "runs/detect/train-3",
    "Model_4": "runs/detect/train-4",
    "Model_5": "runs/detect/train-5",
    "Model_6": "runs/detect/train-7",
}

# 2. Setup the matplotlib figures (Two plots side-by-side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 3. Loop through each model directory, read results.csv, and plot
for model_name, dir_path in model_dirs.items():
    csv_path = os.path.join(dir_path, "results.csv")

    if os.path.exists(csv_path):
        # Read CSV and strip spaces from column headers
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        epochs = df["epoch"]

        # Plot mAP50(B)
        if "metrics/mAP50(B)" in df.columns:
            ax1.plot(
                epochs,
                df["metrics/mAP50(B)"],
                label=f"{model_name}",
                linewidth=2,
            )
        else:
            print(f"Column 'metrics/mAP50(B)' missing in {model_name}")

        # Plot mAP50-95(B)
        if "metrics/mAP50-95(B)" in df.columns:
            ax2.plot(
                epochs,
                df["metrics/mAP50-95(B)"],
                label=f"{model_name}",
                linewidth=2,
            )
        else:
            print(f"Column 'metrics/mAP50-95(B)' missing in {model_name}")
    else:
        print(f"Warning: {csv_path} not found. Skipping {model_name}.")

# 4. Format the mAP50 Plot
ax1.set_title("mAP@0.50 Comparison", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("mAP50", fontsize=12)
ax1.set_ylim(0, 1.05)  # mAP ranges from 0 to 1
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=10, loc="lower right")

# 5. Format the mAP50-95 Plot
ax2.set_title("mAP@0.50:0.95 Comparison", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("mAP50-95", fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=10, loc="lower right")

plt.tight_layout()

# Save the combined metrics plot
plt.savefig("yolo_map_comparison.png", dpi=300)
plt.show()