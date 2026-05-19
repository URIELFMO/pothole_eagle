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

# 2. Setup the matplotlib figures
# We will create two plots: one for Training Loss and one for Validation Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 3. Loop through each model directory, read results.csv, and plot
for model_name, dir_path in model_dirs.items():
    csv_path = os.path.join(dir_path, "results.csv")

    if os.path.exists(csv_path):
        # Read the CSV and strip spaces from column names
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # YOLOv8/v11/v26 names columns as: train/box_loss, train/cls_loss, train/dfl_loss
        # We calculate the Total Loss by summing these components up
        train_loss = (
            df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
        )
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

        epochs = df["epoch"]

        # Plot Training Loss
        ax1.plot(epochs, train_loss, label=f"{model_name} (Train)", linewidth=2)

        # Plot Validation Loss
        ax2.plot(
            epochs,
            val_loss,
            label=f"{model_name} (Val)",
            linestyle="--",
            linewidth=2,
        )
    else:
        print(f"Warning: {csv_path} not found. Skipping {model_name}.")

# 4. Format the plots
# Training Loss Plot Subtleties
ax1.set_title("Total Training Loss Comparison", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Total Loss", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=10)

# Validation Loss Plot Subtleties
ax2.set_title(
    "Total Validation Loss Comparison", fontsize=14, fontweight="bold"
)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Total Loss", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()

# Save the combined plot
plt.savefig("yolo_loss_comparison.png", dpi=300)
plt.show()