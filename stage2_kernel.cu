__global__ void evaluate_G_kernel(
    float* results_G,       // Output array for storing G results
    const float* results_F, // Input array for F results
    float scale_x,          // Scaling coefficient for x
    float scale_w,          // Scaling coefficient for w
    int num_variations_x,   // Number of variations for x
    int num_variations_y,   // Number of variations for y
    int num_variations_z,   // Number of variations for z
    int num_variations_w    // Number of variations for w
) {
    int idx_G = blockIdx.x * blockDim.x + threadIdx.x;
    int num_variations_xyz = num_variations_x * num_variations_y * num_variations_z;

    int w_index = idx_G / num_variations_xyz;
    float w = scale_w * w_index;

    int xyz_index = idx_G % num_variations_xyz;

    int num_variations_y_z = num_variations_y * num_variations_z;
    int x_index = xyz_index / num_variations_y_z;
    // int y_index = (xyz_index % num_variations_y_z) / num_variations_z; // Not needed for G(F, x, w)
    // int z_index = xyz_index % num_variations_z; // Not needed for G(F, x, w)

    float x = scale_x * x_index;

    float f_value = results_F[xyz_index];

    // Example scalar function G(F, x, w)
    float result_G = f_value - pow(x - 0.5f, 2.0f) - pow(w - 900.0f, 2.0f);

    results_G[idx_G] = result_G;
}