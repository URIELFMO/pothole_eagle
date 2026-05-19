from ultralytics import YOLO
import cv2
from tqdm import tqdm
import subprocess, os

# Explicitly set to CPU
model = YOLO("runs/detect/train-30/weights/best.pt")

def run_inference_video(model, input_path, output_path, conf=0.25, iou=0.45):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use raw strings or join paths to avoid backslash errors
    temp = output_path.replace(".MP4", "_temp.mp4")
    print(temp)
  
    writer = cv2.VideoWriter(
        temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
  
    # Using 'stream=True' is more memory-efficient for CPUs
    with tqdm(total=total, desc="Processing (CPU)", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run inference
            results = model(frame, conf=conf, iou=iou, verbose=False)
            
            writer.write(results[0].plot())
            pbar.update(1)
  
    cap.release()
    writer.release()
  
    # Re-encode for compatibility
    subprocess.run([
        "ffmpeg", "-y", "-i", temp,
        "-vcodec", "libx264", "-crf", "28", "-preset", "faster", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(temp)
    print(f"Saved to: {output_path}")

# Note the 'r' before the string to handle Windows paths correctly
videof = r"videoimages\GX011225.MP4"
base, ext = os.path.splitext(videof)
videoout = f"{base}_processed{ext}"
print(videoout)

run_inference_video(model, videof, videoout)