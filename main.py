import argparse
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from src.inference.predict import run_prediction
from src.inference.yolo_cam import run_cam
from scripts.train import train

def main():
    parser = argparse.ArgumentParser(description="Pothole GDL Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the YOLO model")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on a video")
    predict_parser.add_argument("--video", type=str, help="Path to input video")
    predict_parser.add_argument("--model", type=str, help="Path to model weights")

    # CAM command
    cam_parser = subparsers.add_parser("cam", help="Run YOLO CAM on a video")
    cam_parser.add_argument("--video", type=str, help="Path to input video")
    cam_parser.add_argument("--model", type=str, help="Path to model weights")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        kwargs = {}
        if args.video: kwargs['video_path'] = Path(args.video)
        if args.model: kwargs['model_path'] = Path(args.model)
        run_prediction(**kwargs)
    elif args.command == "cam":
        kwargs = {}
        if args.video: kwargs['video_path'] = Path(args.video)
        if args.model: kwargs['model_path'] = Path(args.model)
        run_cam(**kwargs)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
