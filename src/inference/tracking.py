import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from src.config import TRAINED_MODEL_PATH, VIDEOS_DIR, RUNS_DIR, DEFAULT_IMGSZ, DEFAULT_CONF

def run_tracking(
    model_path=TRAINED_MODEL_PATH,
    video_path=VIDEOS_DIR / "GX011256.MP4",
    output_base_dir=RUNS_DIR / "detect" / "video_tracking_custom",
    imgsz=DEFAULT_IMGSZ,
    conf=0.50,
    frame_stride=6
):
    video_name = Path(video_path).stem
    video_output_dir = output_base_dir / video_name
    images_dir = video_output_dir / "images"
    frames_dir = video_output_dir / "frames"
    labels_dir = video_output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = video_output_dir / "tracked_output.mp4"

    # Load model
    model = YOLO(str(model_path))

    # Open video info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_fps = max(fps / frame_stride, 1)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height)
    )

    # Track
    results = model.track(
        source=str(video_path),
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        save=False,
        save_txt=False,
        save_conf=True,
        device=0,
        imgsz=imgsz,
        conf=conf,
        vid_stride=frame_stride,
        verbose=False
    )

    processed_count = 0
    saved_count = 0

    for result in results:
        processed_count += 1
        if result.boxes is None or len(result.boxes) == 0:
            continue

        saved_count += 1
        original_frame_number = processed_count * frame_stride
        frame_name = f"frame_{original_frame_number:06d}"

        # Draw boxes with tracking IDs
        annotated_frame = result.plot()

        # Save frame
        cv2.imwrite(str(images_dir / f"{frame_name}.jpg"), result.orig_img)
        # Save annotated frame
        cv2.imwrite(str(frames_dir / f"{frame_name}.jpg"), annotated_frame)
        # Save labels
        result.save_txt(str(labels_dir / f"{frame_name}.txt"), save_conf=True)
        # Write output video
        writer.write(annotated_frame)

    writer.release()
    print(f"Done. Processed: {processed_count}, Saved: {saved_count}, Output: {output_video_path}")

if __name__ == '__main__':
    run_tracking()
