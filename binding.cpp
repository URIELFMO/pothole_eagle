#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor trapezoid_filter_cuda(torch::Tensor boxes, torch::Tensor trapezoid);

// C++ wrapper with validation
torch::Tensor filter_trapezoid(torch::Tensor boxes, torch::Tensor trapezoid) {
    // Check devices
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(trapezoid.is_cuda(), "trapezoid must be a CUDA tensor");

    // Check types
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32, "boxes must be float32");
    TORCH_CHECK(trapezoid.scalar_type() == torch::kFloat32, "trapezoid must be float32");

    // Check shapes
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N, 4)");
    TORCH_CHECK(trapezoid.dim() == 2 && trapezoid.size(0) == 4 && trapezoid.size(1) == 2, "trapezoid must be (4, 2)");

    // Check continuity
    TORCH_CHECK(boxes.is_contiguous(), "boxes must be contiguous");
    TORCH_CHECK(trapezoid.is_contiguous(), "trapezoid must be contiguous");

    return trapezoid_filter_cuda(boxes, trapezoid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_trapezoid", &filter_trapezoid, "Trapezoid ROI filter for bounding boxes (CUDA)");
}
