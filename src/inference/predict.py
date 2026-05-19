import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from src.config import TRAINED_MODEL_PATH, VIDEOS_DIR, RUNS_DIR, DEFAULT_STRIDE, DEFAULT_IMGSZ

def run_prediction(
    model_path=TRAINED_MODEL_PATH,
    video_path=VIDEOS_DIR / "GX011259_SHORT.mp4",
    output_base_dir=RUNS_DIR / "detect" / "video_predictions",
    frame_stride=DEFAULT_STRIDE,
    imgsz=DEFAULT_IMGSZ
):
    # Load model
    model = YOLO(str(model_path))

    # Output folders
    video_name = Path(video_path).stem
    frames_dir = output_base_dir / video_name / "frames"
    annotated_dir = output_base_dir / video_name / "annotated_frames"
    labels_dir = output_base_dir / video_name / "labels"

    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(video_path),
        stream=True,
        save=False,
        save_txt=False,
        save_conf=True,
        project=str(RUNS_DIR / "detect"),
        name="video_predictions",
        exist_ok=True,
        device=0,
        vid_stride=frame_stride,
        imgsz=imgsz
    )

    processed_count = 0
    saved_count = 0

    for result in results:
        processed_count += 1

        # Save only frames with detection
        if result.boxes is not None and len(result.boxes) > 0:
            saved_count += 1

            # This is the approximate original frame number
            original_frame_number = processed_count * frame_stride
            frame_name = f"frame_{original_frame_number:06d}"

            raw_frame_path = frames_dir / f"{frame_name}.jpg"
            annotated_frame_path = annotated_dir / f"{frame_name}.jpg"
            txt_path = labels_dir / f"{frame_name}.txt"

            # Original frame
            frame = result.orig_img

            # Annotated frame
            annotated_frame = result.plot()

            # Save raw frame
            cv2.imwrite(str(raw_frame_path), frame)

            # Save annotated frame
            cv2.imwrite(str(annotated_frame_path), annotated_frame)

            # Save YOLO txt labels
            result.save_txt(str(txt_path), save_conf=True)

    print(f"Processed sampled frames: {processed_count}")
    print(f"Saved detection frames: {saved_count}")
    print(f"Output folder: {output_base_dir / video_name}")

if __name__ == '__main__':
    run_prediction()
