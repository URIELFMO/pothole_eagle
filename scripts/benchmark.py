from ultralytics.utils.benchmarks import benchmark
import pandas as pd

# This will benchmark your specific YOLO26 model
# It evaluates speed and accuracy across different export formats
results = benchmark(
    model='runs/detect/train-7/weights/best.pt', 
    data='data_own.yaml', 
    imgsz=640, 
    half=True,   # Use FP16 (essential for Jetson/Radeon performance)
    device=0     # 0 for your GPU, or 'cpu'
)

# The function returns a pandas DataFrame you can inspect or save
df = pd.DataFrame(results)

# Now you can save it
df.to_csv("benchmark_results_gdl.csv")