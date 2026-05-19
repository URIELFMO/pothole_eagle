import torch
import cv2
import numpy as np
from ultralytics import YOLO
import trapezoid_filter
from tqdm import tqdm

def main():
    # 1. Initialize YOLO model
    # Using .engine for high performance
    model_path = "runs/detect/train-7/weights/best.engine"
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"Model {model_path} not found, falling back to .pt...")
        model_path = "runs/detect/train-7/weights/best.pt"
        try:
            model = YOLO(model_path)
        except Exception:
            pass

    # 2. Define Trapezoid ROI (Clockwise order)
    # Enlarged coordinates for better coverage
    trapezoid = torch.tensor([
        [100, 1080],  # Bottom Left
        [1820, 1080], # Bottom Right
        [1300, 400],  # Top Right
        [620, 400]    # Top Left
    ], dtype=torch.float32, device="cuda")

    # 3. Load Source
    video_path = "videos/GX011259_SHORT.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    # Get video properties for VideoWriter
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Initialize Video Writer
    output_path = "output_filtered_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")

    # 5. Process loop
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Run Inference
        results = model(frame, device=0, verbose=False)
        
        # Get raw boxes
        if len(results[0].boxes) > 0:
            # Ensure CUDA and Contiguous for the extension
            boxes = results[0].boxes.xyxy.cuda().contiguous()
            
            # Use the CUDA Extension to filter boxes
            mask = trapezoid_filter.filter_trapezoid(boxes, trapezoid)
            
            # Draw ALL boxes with different colors
            boxes_np = boxes.cpu().numpy().astype(int)
            mask_np = mask.cpu().numpy()

            for i, box in enumerate(boxes_np):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if mask_np[i]:
                    # INSIDE ROI: Green
                    color = (0, 255, 0)
                    thickness = 2
                    label = "Inside"
                else:
                    # OUTSIDE ROI: Red
                    color = (0, 0, 255)
                    thickness = 1
                    label = "Outside"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # Center point in Red

        # Visualization: Always Draw ROI (Yellow)
        roi_pts = trapezoid.cpu().numpy().astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # Write frame to output
        out.write(frame)

    # 6. Cleanup
    cap.release()
    out.release()
    print(f"Processing complete. Saved video to {output_path}")

if __name__ == "__main__":
    main()
