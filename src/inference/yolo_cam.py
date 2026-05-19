import cv2
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.config import TRAINED_MODEL_PATH, VIDEOS_DIR, RUNS_DIR, DEFAULT_IMGSZ, DEFAULT_CONF, DEFAULT_STRIDE

def run_cam(
    model_path=TRAINED_MODEL_PATH,
    video_path=VIDEOS_DIR / "GX011259_SHORT.mp4",
    output_base_dir=RUNS_DIR / "detect" / "video_cam",
    imgsz=DEFAULT_IMGSZ,
    conf=DEFAULT_CONF,
    frame_stride=DEFAULT_STRIDE,
    save_only_detections=True,
    save_frames=True,
    save_txt=True
):
    # Output setup
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_base_dir / "yolo_cam_output.mp4"
    labels_dir = output_base_dir / "labels"
    frames_dir = output_base_dir / "frames_with_cam"

    labels_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    model = YOLO(str(model_path))

    # Hook target layer for CAM
    activations = {}
    
    # Usually the second-to-last model layer is a good starting point before Detect head.
    target_layer = model.model.model[-2]

    def forward_hook(module, input, output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        activations["value"] = output.detach()

    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Helper: create heatmap
    def make_cam_heatmap(activation_tensor, original_frame):
        fmap = activation_tensor[0].float().cpu()
        cam = torch.relu(fmap.mean(dim=0)).numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = np.uint8(255 * cam)
        h, w = original_frame.shape[:2]
        cam = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        return heatmap

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = max(source_fps / frame_stride, 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        output_fps,
        (source_width, source_height)
    )

    frame_number = 0
    processed_count = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if frame_number % frame_stride != 0:
            continue

        processed_count += 1
        results = model.predict(source=frame, device=0, imgsz=imgsz, conf=conf, verbose=False, save=False)
        result = results[0]
        has_detections = result.boxes is not None and len(result.boxes) > 0

        if save_only_detections and not has_detections:
            continue

        saved_count += 1
        annotated_frame = result.plot()

        if "value" in activations:
            heatmap = make_cam_heatmap(activations["value"], frame)
            cam_overlay = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)
            final_frame = cv2.addWeighted(cam_overlay, 0.70, annotated_frame, 0.30, 0)
        else:
            final_frame = annotated_frame

        frame_name = f"frame_{frame_number:06d}"
        if save_frames:
            cv2.imwrite(str(frames_dir / f"{frame_name}.jpg"), final_frame)
        if save_txt and has_detections:
            result.save_txt(str(labels_dir / f"{frame_name}.txt"), save_conf=True)
        writer.write(final_frame)

    cap.release()
    writer.release()
    hook_handle.remove()

    print("Done")
    print(f"Processed sampled frames: {processed_count}")
    print(f"Saved detection frames: {saved_count}")
    print(f"Output video: {output_video_path}")

if __name__ == '__main__':
    run_cam()
