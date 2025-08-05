__global__ void evaluate_F_kernel(
    float* results_F,        // Output array for storing F results
    float scale_x,         // Scaling coefficient for x
    float scale_y,         // Scaling coefficient for y
    float scale_z,         // Scaling coefficient for z
    int num_variations_x,  // Number of variations for x
    int num_variations_y,  // Number of variations for y
    int num_variations_z   // Number of variations for z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_variations_y_z = num_variations_y * num_variations_z;

    int x_index = idx / num_variations_y_z;
    int y_index = (idx % num_variations_y_z) / num_variations_z;
    int z_index = idx % num_variations_z;

    float x = scale_x * x_index;
    float y = scale_y * y_index;
    float z = scale_z * z_index;

    // Example scalar function F(x, y, z)
    float result_F = -pow(x - 0.5f, 2.0f) - pow(y - 2.0f, 2.0f) - pow(z - 5.0f, 2.0f);

    results_F[idx] = result_F;
}