#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void trapezoid_filter_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ trapezoid,
    bool* __restrict__ mask,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load box: [x1, y1, x2, y2]
    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float x2 = boxes[idx * 4 + 2];
    float y2 = boxes[idx * 4 + 3];

    // Compute center point
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;

    // Trapezoid points (p0, p1, p2, p3) in clockwise order
    // Edge function: (x_next - x_curr) * (y_p - y_curr) - (y_next - y_curr) * (x_p - x_curr)
    
    bool inside = true;
    for (int i = 0; i < 4; ++i) {
        float x_curr = trapezoid[i * 2 + 0];
        float y_curr = trapezoid[i * 2 + 1];
        float x_next = trapezoid[((i + 1) % 4) * 2 + 0];
        float y_next = trapezoid[((i + 1) % 4) * 2 + 1];

        float vx = x_next - x_curr;
        float vy = y_next - y_curr;
        float ux = cx - x_curr;
        float uy = cy - y_curr;

        // In screen coordinates (y-down), for clockwise points:
        // Inside points will have a negative cross product with the edge vectors.
        // If cross product is positive, the point is outside.
        if ((vx * uy - vy * ux) > 0.0f) {
            inside = false;
            break;
        }
    }

    mask[idx] = inside;
}

torch::Tensor trapezoid_filter_cuda(torch::Tensor boxes, torch::Tensor trapezoid) {
    const int n = boxes.size(0);
    auto mask = torch::empty({n}, boxes.options().dtype(torch::kBool));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    trapezoid_filter_kernel<<<blocks, threads>>>(
        boxes.data_ptr<float>(),
        trapezoid.data_ptr<float>(),
        mask.data_ptr<bool>(),
        n
    );

    return mask;
}
