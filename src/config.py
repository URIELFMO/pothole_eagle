from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR  # Assuming train/test/valid are at root
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIDEOS_DIR = BASE_DIR / "videos"

# Model Paths
DEFAULT_MODEL_PATH = MODELS_DIR / "yolo26s.pt"
TRAINED_MODEL_PATH = RUNS_DIR / "detect" / "train-7" / "weights" / "best.pt"

# Data Paths
DATA_YAML = CONFIGS_DIR / "data.yaml"

# Inference Config
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25
DEFAULT_STRIDE = 30

def ensure_dirs():
    """Ensure all required directories exist."""
    for dir_path in [MODELS_DIR, RUNS_DIR, OUTPUTS_DIR, VIDEOS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    ensure_dirs()
