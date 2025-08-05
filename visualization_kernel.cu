__device__ unsigned char map_value_to_gray(float value, float min_value, float max_value) {
    float normalized_value = (value - min_value) / (max_value - min_value);
    if (normalized_value < 0.0f) normalized_value = 0.0f;
    if (normalized_value > 1.0f) normalized_value = 1.0f;
    return (unsigned char)(normalized_value * 255.0f);
}

__global__ void visualize_kernel(
    unsigned char* image_output, // Output array for the image (grayscale)
    const float* results_G,      // Input array for G results
    float min_result,            // Minimum result value for color mapping
    float max_result,            // Maximum result value for color mapping
    int num_variations_x,
    int num_variations_y,
    int num_variations_z,
    int num_variations_w,
    int padding_pixels         // Padding between inner grids
) {
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;

    int inner_grid_width = num_variations_z;
    int inner_grid_height = num_variations_w;
    int tiled_grid_width = inner_grid_width + padding_pixels;
    int tiled_grid_height = inner_grid_height + padding_pixels;

    int total_img_width = tiled_grid_width * num_variations_x + padding_pixels;
    int total_img_height = tiled_grid_height * num_variations_y + padding_pixels;

    if (img_x >= total_img_width || img_y >= total_img_height) {
        return; // Out of bounds
    }

    bool is_padding = false;
    if ((img_x % tiled_grid_width) < padding_pixels ||
        (img_y % tiled_grid_height) < padding_pixels) {
        is_padding = true;
    }

    if (is_padding) {
        image_output[img_y * total_img_width + img_x] = 0; // Black padding
        return;
    }

    int tile_x_index = img_x / tiled_grid_width;
    int tile_y_index = img_y / tiled_grid_height;

    int inner_x = (img_x % tiled_grid_width) - padding_pixels; // z index
    int inner_y = (img_y % tiled_grid_height) - padding_pixels; // w index

    if (inner_x < 0 || inner_y < 0) {
         return;
    }

    // Map back to 4D parameter indices (x, y, z, w) based on Stage 2 kernel indexing
    int x_index = tile_x_index;
    int y_index = tile_y_index;
    int z_index = inner_x;
    int w_index = inner_y;

    int num_variations_y_z = num_variations_y * num_variations_z;
    int num_variations_xyz = num_variations_x * num_variations_y * num_variations_z;
    int results_index = w_index * num_variations_xyz +
                        x_index * num_variations_y_z +
                        y_index * num_variations_z +
                        z_index;

    float result = results_G[results_index];
    unsigned char gray_color = map_value_to_gray(result, min_result, max_result);
    image_output[img_y * total_img_width + img_x] = gray_color;
}