import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.config import TRAINED_MODEL_PATH, VIDEOS_DIR, RUNS_DIR, DEFAULT_STRIDE, DEFAULT_IMGSZ, DEFAULT_CONF

def draw_transparent_trapezoid(frame, points, color=(0, 255, 255), alpha=0.25):
    overlay = frame.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
    x_label, y_label = pts[0]
    cv2.putText(frame, "ROI", (x_label + 10, y_label + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def create_trapezoid_mask(frame_shape, points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def point_inside_polygon(x, y, polygon_points):
    pts = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def run_roi_analysis(
    model_path=TRAINED_MODEL_PATH,
    video_path=VIDEOS_DIR / "GX011259_SHORT.mp4",
    output_base_dir=RUNS_DIR / "detect" / "video_tracking_trapezoid_roi",
    frame_stride=DEFAULT_STRIDE,
    imgsz=640, # ROI usually uses smaller imgsz for speed
    conf=DEFAULT_CONF
):
    video_name = Path(video_path).stem
    roi_dir = output_base_dir / f"{video_name}_analisis"
    roi_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = roi_dir / "tracked_trapezoid_roi_output.mp4"

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    full_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = max(fps / frame_stride, 1)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (full_width, full_height)
    )

    center_x = full_width // 2
    top_y = 200
    bottom_y = full_height
    top_width = 400
    bottom_width = full_width

    trapezoid_points = [
        (center_x - top_width // 2, top_y),
        (center_x + top_width // 2, top_y),
        (center_x + bottom_width // 2, bottom_y),
        (center_x - bottom_width // 2, bottom_y)
    ]
    trapezoid_points = [(max(0, min(x, full_width - 1)), max(0, min(y, full_height - 1))) for x, y in trapezoid_points]

    xs = [p[0] for p in trapezoid_points]
    ys = [p[1] for p in trapezoid_points]
    crop_x1, crop_x2 = max(0, min(xs)), min(full_width, max(xs))
    crop_y1, crop_y2 = max(0, min(ys)), min(full_height, max(ys))

    frame_number = 0
    processed_count = 0
    detections_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_number += 1
        if frame_number % frame_stride != 0: continue
        processed_count += 1

        final_frame = draw_transparent_trapezoid(frame.copy(), trapezoid_points)
        full_mask = create_trapezoid_mask(frame.shape, trapezoid_points)
        crop_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = full_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        masked_crop = cv2.bitwise_and(crop_frame, crop_frame, mask=crop_mask)

        results = model.track(source=masked_crop, persist=True, tracker="bytetrack.yaml", device=0, imgsz=imgsz, conf=conf, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

            frame_has_valid_detection = False
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, x2, y1, y2 = x1 + crop_x1, x2 + crop_x1, y1 + crop_y1, y2 + crop_y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if not point_inside_polygon(cx, cy, trapezoid_points): continue
                frame_has_valid_detection = True
                label = f"ID {int(track_ids[i]) if track_ids is not None else ''} cls {int(classes[i])} {float(confs[i]):.2f}"
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(final_frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if frame_has_valid_detection: detections_count += 1
        writer.write(final_frame)

    cap.release()
    writer.release()
    print(f"Done. Processed: {processed_count}, Detections: {detections_count}, Output: {output_video_path}")

if __name__ == '__main__':
    run_roi_analysis()
