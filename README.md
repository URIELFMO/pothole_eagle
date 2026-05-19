# Trapezoid ROI Filter CUDA Extension

A high-performance PyTorch CUDA extension to filter bounding boxes based on a trapezoidal Region of Interest (ROI).

## Installation

Ensure you have PyTorch and a CUDA toolkit installed. Then, run:

```bash
cd /home/morau/pothole_eagle
pip install -e .
```

## Usage

The extension provides a `filter_trapezoid` function that takes bounding boxes and a 4-point trapezoid, returning a boolean mask.

```python
import torch
import trapezoid_filter

# boxes: (N, 4) tensor [x1, y1, x2, y2] on CUDA
# trapezoid: (4, 2) tensor of points in clockwise order on CUDA
mask = trapezoid_filter.filter_trapezoid(boxes, trapezoid)

filtered_boxes = boxes[mask]
```

## Files
- `trapezoid_filter.cu`: CUDA kernel implementation for point-in-quadrilateral test.
- `trapezoid_filter.cpp`: PyTorch C++ bindings and input validation.
- `setup.py`: Compilation script.
- `example.py`: Full example integrating with YOLO.
