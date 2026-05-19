import torch
from ultralytics import YOLO
from src.config import DEFAULT_MODEL_PATH, DATA_YAML

def train():
    # NVIDIA CUDA optimization
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"Using NVIDIA GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("CUDA GPU not detected. Using CPU.")

    # Load model
    model = YOLO(str(DEFAULT_MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=200,
        patience=40,
        imgsz=640,
        device=device,
        batch=-1,
        amp=True,        
        # Data loading
        workers=4,
        cache=True,
        close_mosaic=10,
        save=True
    )

if __name__ == '__main__':
    train()
