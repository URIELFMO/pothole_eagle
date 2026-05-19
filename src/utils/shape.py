import matplotlib.pyplot as plt
import numpy as np

# =========================
# Video/frame dimensions
# Same as your video
# =========================
full_width = 1920
full_height = 1080

# =========================
# Isosceles trapezoid settings
# Same logic as YOLO ROI code
# =========================
center_x = full_width // 2

top_y = 500
bottom_y = full_height

top_width = 800          # narrow top side
bottom_width = full_width # wide bottom side

# =========================
# Trapezoid points
# Order:
# top-left, top-right, bottom-right, bottom-left
# =========================
trapezoid_points = np.array([
    [center_x - top_width // 2, top_y],          # top-left
    [center_x + top_width // 2, top_y],          # top-right
    [center_x + bottom_width // 2, bottom_y],    # bottom-right
    [center_x - bottom_width // 2, bottom_y]     # bottom-left
])

# Clamp points to frame boundaries
trapezoid_points[:, 0] = np.clip(trapezoid_points[:, 0], 0, full_width)
trapezoid_points[:, 1] = np.clip(trapezoid_points[:, 1], 0, full_height)

# Close polygon for line plotting
closed_points = np.vstack([trapezoid_points, trapezoid_points[0]])

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(12, 7))

# Draw full video frame
ax.set_xlim(0, full_width)
ax.set_ylim(full_height, 0)  # invert Y axis to match image/video coordinates

# Background frame area
ax.add_patch(
    plt.Rectangle(
        (0, 0),
        full_width,
        full_height,
        fill=False,
        edgecolor="black",
        linewidth=2,
        label="Video frame 1920x1080"
    )
)

# Draw transparent yellow trapezoid
ax.fill(
    trapezoid_points[:, 0],
    trapezoid_points[:, 1],
    color="yellow",
    alpha=0.35,
    label="Isosceles trapezoid ROI"
)

# Draw trapezoid border
ax.plot(
    closed_points[:, 0],
    closed_points[:, 1],
    color="orange",
    linewidth=3
)

# Draw points and labels
labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]

for point, label in zip(trapezoid_points, labels):
    x, y = point
    ax.scatter(x, y, color="red", s=60)
    ax.text(
        x + 15,
        y - 15,
        f"{label}\n({int(x)}, {int(y)})",
        color="red",
        fontsize=10,
        weight="bold"
    )

# Draw center line
ax.axvline(center_x, color="blue", linestyle="--", linewidth=1.5, label="Center line")

# Grid and labels
ax.set_title("Isosceles Trapezoid ROI Visualization - 1920x1080", fontsize=14)
ax.set_xlabel("X pixels")
ax.set_ylabel("Y pixels")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()