#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for RGB to grayscale conversion
__global__ void grayscale_cuda(float *rgb_image, float *gray_image, int width, int height)
{
    // Calculate the global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within image boundaries
    if (x < width && y < height)
    {
        // Calculate the index in the grayscale image
        int gray_idx = y * width + x;

        // Calculate the corresponding indices in the RGB image
        // RGB image is in NCHW format (depth, width, height)
        int r_idx = 0 * width * height + y * width + x; // Red channel
        int g_idx = 1 * width * height + y * width + x; // Green channel
        int b_idx = 2 * width * height + y * width + x; // Blue channel

        // Apply standard RGB to grayscale conversion formula
        // Luminance = 0.299 * R + 0.587 * G + 0.114 * B
        gray_image[gray_idx] = 0.299f * rgb_image[r_idx] +
                               0.587f * rgb_image[g_idx] +
                               0.114f * rgb_image[b_idx];
    }
}

// CUDA kernel for Gaussian smoothing
__global__ void gaussian_smooth_cuda(float *input_image, float *output_image,
                                     float *kernel, int kernel_size,
                                     int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        float kernel_sum = 0.0f;

        // Half the size of the kernel (rounded down)
        int half_kernel = kernel_size / 2;

        // Apply the kernel
        for (int ky = -half_kernel; ky <= half_kernel; ky++)
        {
            for (int kx = -half_kernel; kx <= half_kernel; kx++)
            {
                // Calculate image coordinates with boundary check
                int img_x = min(max(x + kx, 0), width - 1);
                int img_y = min(max(y + ky, 0), height - 1);

                // Calculate kernel index
                int k_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);

                // Get pixel value and kernel coefficient
                float pixel = input_image[img_y * width + img_x];
                float k_value = kernel[k_idx];

                // Accumulate weighted sum
                sum += pixel * k_value;
                kernel_sum += k_value;
            }
        }

        // Normalize by the sum of kernel values to ensure brightness is preserved
        if (kernel_sum > 0)
        {
            output_image[y * width + x] = sum / kernel_sum;
        }
        else
        {
            output_image[y * width + x] = input_image[y * width + x];
        }
    }
}

// Function to create a Gaussian kernel
void create_gaussian_kernel(float *kernel, int kernel_size, float sigma)
{
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    // Calculate Gaussian kernel values
    for (int y = -half_kernel; y <= half_kernel; y++)
    {
        for (int x = -half_kernel; x <= half_kernel; x++)
        {
            int idx = (y + half_kernel) * kernel_size + (x + half_kernel);
            // Gaussian function: G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
            kernel[idx] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[idx];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size * kernel_size; i++)
    {
        kernel[i] /= sum;
    }
}

// Function to handle the RGB to grayscale conversion using CUDA
void convert_to_grayscale(float *rgb_image, float *gray_image, int width, int height)
{
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    grayscale_cuda<<<gridSize, blockSize>>>(rgb_image, gray_image, width, height);

    // Synchronize to make sure the kernel execution is complete
    cudaDeviceSynchronize();
}

// Function to apply Gaussian smoothing using CUDA
void apply_gaussian_smooth(float *input_image, float *output_image, int width, int height,
                           int kernel_size, float sigma)
{
    // Allocate memory for the Gaussian kernel
    float *kernel;
    size_t kernel_mem_size = kernel_size * kernel_size * sizeof(float);
    cudaMallocManaged(&kernel, kernel_mem_size);

    // Create the Gaussian kernel
    create_gaussian_kernel(kernel, kernel_size, sigma);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    gaussian_smooth_cuda<<<gridSize, blockSize>>>(input_image, output_image,
                                                  kernel, kernel_size, width, height);

    // Synchronize to make sure the kernel execution is complete
    cudaDeviceSynchronize();

    // Free the kernel memory
    cudaFree(kernel);
}

int main()
{
    int width = 1024;
    int height = 768;
    size_t rgb_size = 3 * width * height * sizeof(float);
    size_t gray_size = width * height * sizeof(float);

    // Allocate managed memory for input and output
    float *rgb_image, *gray_image, *smoothed_image;
    cudaMallocManaged(&rgb_image, rgb_size);
    cudaMallocManaged(&gray_image, gray_size);
    cudaMallocManaged(&smoothed_image, gray_size);

    // Initialize the RGB image (this would be your input data)
    // For example purposes, let's fill it with some values
    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = c * width * height + y * width + x;
                rgb_image[idx] = (float)(c + x + y) / (3 + width + height); // Just a sample value
            }
        }
    }

    // Convert the RGB image to grayscale
    convert_to_grayscale(rgb_image, gray_image, width, height);

    // Apply Gaussian smoothing to the grayscale image
    // Parameters: kernel size=5, sigma=1.0
    int kernel_size = 5;
    float sigma = 1.0f;
    apply_gaussian_smooth(gray_image, smoothed_image, width, height, kernel_size, sigma);

    // Verify the output (print a small sample of the smoothed image)
    printf("Sample smoothed grayscale values:\n");
    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            printf("%0.3f ", smoothed_image[y * width + x]);
        }
        printf("\n");
    }

    // Free the allocated memory
    cudaFree(rgb_image);
    cudaFree(gray_image);
    cudaFree(smoothed_image);

    return 0;
}