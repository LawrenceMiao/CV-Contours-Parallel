#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include "clockcycle.h" // POWER9 cycle counter

// Structure to represent an RGB image
typedef struct
{
    int width;
    int height;
    unsigned char *data; // For RGB, each pixel has 3 bytes
} RGBImage;

// Structure to represent a grayscale image
typedef struct
{
    int width;
    int height;
    unsigned char *data; // For grayscale, each pixel has 1 byte
} GrayImage;

// Function to read PNG file into RGB image
RGBImage readPNG(const char *filename)
{
    RGBImage img = {0};
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Could not open file %s\n", filename);
        return img;
    }

    // Check if it's a PNG file
    unsigned char header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        printf("Error: %s is not a PNG file\n", filename);
        fclose(fp);
        return img;
    }

    // Initialize PNG read structure
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        printf("Error: Could not create PNG read structure\n");
        fclose(fp);
        return img;
    }

    // Initialize PNG info structure
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        printf("Error: Could not create PNG info structure\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return img;
    }

    // Setup error handling
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error during PNG file reading\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return img;
    }

    // Initialize PNG I/O
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    // Read PNG info
    png_read_info(png_ptr, info_ptr);

    // Get image info
    img.width = png_get_image_width(png_ptr, info_ptr);
    img.height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Set up transformations to ensure RGB output
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // Update the info structure
    png_read_update_info(png_ptr, info_ptr);

    // Allocate memory for the image data
    img.data = (unsigned char *)malloc(img.width * img.height * 3);
    if (!img.data)
    {
        printf("Error: Could not allocate memory for image data\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        img.width = 0;
        img.height = 0;
        return img;
    }

    // Read the image data
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * img.height);
    for (int y = 0; y < img.height; y++)
    {
        row_pointers[y] = img.data + y * img.width * 3;
    }
    png_read_image(png_ptr, row_pointers);
    free(row_pointers);

    // Clean up
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return img;
}

// Function to write grayscale image to PNG file
void writeGrayPNG(const char *filename, const GrayImage *img)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Initialize PNG write structure
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        printf("Error: Could not create PNG write structure\n");
        fclose(fp);
        return;
    }

    // Initialize PNG info structure
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        printf("Error: Could not create PNG info structure\n");
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return;
    }

    // Setup error handling
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error during PNG file writing\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    // Initialize PNG I/O
    png_init_io(png_ptr, fp);

    // Set the image info
    png_set_IHDR(png_ptr, info_ptr, img->width, img->height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Write the PNG info
    png_write_info(png_ptr, info_ptr);

    // Allocate memory for row pointers
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * img->height);
    for (int y = 0; y < img->height; y++)
    {
        row_pointers[y] = img->data + y * img->width;
    }

    // Write the image data
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    // Clean up
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// Function to convert RGB image to binary file
void rgbToBinary(const RGBImage *img, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Write image dimensions
    fwrite(&img->width, sizeof(int), 1, fp);
    fwrite(&img->height, sizeof(int), 1, fp);

    // Write image data
    fwrite(img->data, sizeof(unsigned char), img->width * img->height * 3, fp);

    fclose(fp);
}

// Function to read binary file into RGB image
RGBImage binaryToRGB(const char *filename)
{
    RGBImage img = {0};
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Could not open file %s\n", filename);
        return img;
    }

    // Read image dimensions
    fread(&img.width, sizeof(int), 1, fp);
    fread(&img.height, sizeof(int), 1, fp);

    // Allocate memory for image data
    img.data = (unsigned char *)malloc(img.width * img.height * 3);
    if (!img.data)
    {
        printf("Error: Could not allocate memory for image data\n");
        fclose(fp);
        img.width = 0;
        img.height = 0;
        return img;
    }

    // Read image data
    fread(img.data, sizeof(unsigned char), img.width * img.height * 3, fp);

    fclose(fp);
    return img;
}

// Function to read binary file into grayscale image
GrayImage binaryToGray(const char *filename)
{
    GrayImage img = {0};
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Could not open file %s\n", filename);
        return img;
    }

    // Read image dimensions
    fread(&img.width, sizeof(int), 1, fp);
    fread(&img.height, sizeof(int), 1, fp);

    // Allocate memory for image data
    img.data = (unsigned char *)malloc(img.width * img.height);
    if (!img.data)
    {
        printf("Error: Could not allocate memory for image data\n");
        fclose(fp);
        img.width = 0;
        img.height = 0;
        return img;
    }

    // Read image data
    fread(img.data, sizeof(unsigned char), img.width * img.height, fp);

    fclose(fp);
    return img;
}

// Structure to hold gradient information
typedef struct
{
    int width;
    int height;
    float *magnitude;
    float *direction;
} GradientImage;

// CUDA kernel for RGB to Grayscale conversion
__global__ void rgbToGrayscaleKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int rgbOffset = (y * width + x) * 3;
        int localOffset = (y - startRow) * width + x;

        unsigned char r = d_input[rgbOffset];
        unsigned char g = d_input[rgbOffset + 1];
        unsigned char b = d_input[rgbOffset + 2];

        // Convert to grayscale using luminance formula
        d_output[localOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        // 5x5 Gaussian kernel
        const float kernel[5][5] = {
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}};

        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ky++)
        {
            for (int kx = -2; kx <= 2; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                // Handle boundary conditions - including partition boundaries
                // Pixels outside the partition are treated as if they're outside the image
                if (ix < 0)
                    ix = 0;
                if (ix >= width)
                    ix = width - 1;
                if (iy < startRow)
                    iy = startRow;
                if (iy >= endRow)
                    iy = endRow - 1;

                int inputOffset = (iy - startRow) * width + ix;
                sum += kernel[ky + 2][kx + 2] * d_input[inputOffset];
            }
        }

        d_output[(y - startRow) * width + x] = (unsigned char)sum;
    }
}

// CUDA kernel for Sobel operator in x direction
__global__ void sobelXKernel(unsigned char *d_input, float *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        // 3x3 Sobel X kernel
        const int kernel[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}};

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                // Handle boundary conditions - including partition boundaries
                // Pixels outside the partition are treated as if they're outside the image
                if (ix < 0)
                    ix = 0;
                if (ix >= width)
                    ix = width - 1;
                if (iy < startRow)
                    iy = startRow;
                if (iy >= endRow)
                    iy = endRow - 1;

                int inputOffset = (iy - startRow) * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[(y - startRow) * width + x] = sum;
    }
}

// CUDA kernel for Sobel operator in y direction
__global__ void sobelYKernel(unsigned char *d_input, float *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        // 3x3 Sobel Y kernel
        const int kernel[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}};

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                // Handle boundary conditions - including partition boundaries
                // Pixels outside the partition are treated as if they're outside the image
                if (ix < 0)
                    ix = 0;
                if (ix >= width)
                    ix = width - 1;
                if (iy < startRow)
                    iy = startRow;
                if (iy >= endRow)
                    iy = endRow - 1;

                int inputOffset = (iy - startRow) * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[(y - startRow) * width + x] = sum;
    }
}

// CUDA kernel to compute gradient magnitude and direction
__global__ void gradientKernel(float *d_gradient_x, float *d_gradient_y, float *d_magnitude, float *d_direction, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int local_idx = (y - startRow) * width + x;
        float gx = d_gradient_x[local_idx];
        float gy = d_gradient_y[local_idx];

        // Compute magnitude
        d_magnitude[local_idx] = sqrtf(gx * gx + gy * gy);

        // Compute direction (in radians)
        d_direction[local_idx] = atan2f(gy, gx);
    }
}

// CUDA kernel for non-maximum suppression
__global__ void nonMaxSuppressionKernel(float *d_magnitude, float *d_direction, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int local_idx = (y - startRow) * width + x;
        float mag = d_magnitude[local_idx];
        float dir = d_direction[local_idx];

        // Convert direction to discrete angle (0, 45, 90, 135 degrees)
        dir = dir * 180.0f / M_PI;
        if (dir < 0)
            dir += 180.0f;

        int angle = ((int)(dir + 22.5f) % 180) / 45;

        // Define neighbor offsets based on the angle
        int nx1, ny1, nx2, ny2;

        switch (angle)
        {
        case 0: // 0 degrees (horizontal)
            nx1 = -1;
            ny1 = 0;
            nx2 = 1;
            ny2 = 0;
            break;
        case 1: // 45 degrees (diagonal)
            nx1 = -1;
            ny1 = -1;
            nx2 = 1;
            ny2 = 1;
            break;
        case 2: // 90 degrees (vertical)
            nx1 = 0;
            ny1 = -1;
            nx2 = 0;
            ny2 = 1;
            break;
        case 3: // 135 degrees (diagonal)
            nx1 = -1;
            ny1 = 1;
            nx2 = 1;
            ny2 = -1;
            break;
        default:
            nx1 = 0;
            ny1 = 0;
            nx2 = 0;
            ny2 = 0;
        }

        // Calculate neighbor positions
        int x1 = x + nx1;
        int y1 = y + ny1;
        int x2 = x + nx2;
        int y2 = y + ny2;

        // Ensure neighbors are within partition bounds
        // Boundary conditions - treat partition boundaries as image boundaries
        bool valid1 = (x1 >= 0 && x1 < width && y1 >= startRow && y1 < endRow);
        bool valid2 = (x2 >= 0 && x2 < width && y2 >= startRow && y2 < endRow);

        float mag1 = 0.0f;
        float mag2 = 0.0f;

        if (valid1)
        {
            mag1 = d_magnitude[(y1 - startRow) * width + x1];
        }

        if (valid2)
        {
            mag2 = d_magnitude[(y2 - startRow) * width + x2];
        }

        // Apply non-maximum suppression
        if (mag >= mag1 && mag >= mag2)
        {
            // Keep this pixel only if it's a local maximum
            // Apply thresholding (can be adjusted)
            const float highThreshold = 50.0f;
            if (mag > highThreshold)
            {
                d_output[local_idx] = 255; // Strong edge
            }
            else
            {
                d_output[local_idx] = 0; // Non-edge
            }
        }
        else
        {
            d_output[local_idx] = 0; // Suppress non-maximum
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <input_png> <output_png>\n", argv[0]);
        return 1;
    }

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *inputPNG = argv[1];
    const char *outputPNG = argv[2];
    const char *inputBinary = "input.bin";
    const char *outputBinary = "output.bin";

    // Step 1: Read PNG and convert to binary (only rank 0)
    RGBImage rgbImg = {0};
    int width, height;

    if (rank == 0)
    {
        printf("Rank %d: Reading PNG file %s\n", rank, inputPNG);
        rgbImg = readPNG(inputPNG);
        if (rgbImg.width == 0 || rgbImg.height == 0)
        {
            printf("Error: Failed to read PNG file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Rank %d: Converting PNG to binary %s\n", rank, inputBinary);
        rgbToBinary(&rgbImg, inputBinary);

        width = rgbImg.width;
        height = rgbImg.height;

        // Free the RGB image data since we'll read it again from binary
        free(rgbImg.data);
        rgbImg.data = NULL;
    }

    // Broadcast image dimensions to all ranks
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Make sure rank 0 has written the binary file before other ranks try to read it
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 2: Each rank reads the entire binary image
    printf("Rank %d: Reading binary file %s\n", rank, inputBinary);
    rgbImg = binaryToRGB(inputBinary);
    if (rgbImg.width == 0 || rgbImg.height == 0)
    {
        printf("Rank %d: Error reading binary file\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* <<<  start the timed region  >>> */
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t t0 = 0;

    if (rank == 0)
    {
        t0 = clock_now();
    }

    // Step 3: Determine the partition for this rank
    int rowsPerRank = height / size;
    int extraRows = height % size;

    // Calculate start and end rows for this rank
    int startRow = rank * rowsPerRank + min(rank, extraRows);
    int endRow = startRow + rowsPerRank + (rank < extraRows ? 1 : 0);
    int rowsInPartition = endRow - startRow;

    printf("Rank %d: Processing rows %d to %d (total %d rows)\n", rank, startRow, endRow - 1, rowsInPartition);

    // Allocate memory for grayscale intermediate and final image data - only for our partition
    unsigned char *grayImgPartition = (unsigned char *)malloc(width * rowsInPartition);
    unsigned char *blurredImgPartition = (unsigned char *)malloc(width * rowsInPartition);
    unsigned char *edgesImgPartition = (unsigned char *)malloc(width * rowsInPartition);

    // Allocate memory for gradient information - only for our partition
    float *gradientXPartition = (float *)malloc(width * rowsInPartition * sizeof(float));
    float *gradientYPartition = (float *)malloc(width * rowsInPartition * sizeof(float));
    float *magnitudePartition = (float *)malloc(width * rowsInPartition * sizeof(float));
    float *directionPartition = (float *)malloc(width * rowsInPartition * sizeof(float));

    // Zero out all partitioned images
    memset(grayImgPartition, 0, width * rowsInPartition);
    memset(blurredImgPartition, 0, width * rowsInPartition);
    memset(edgesImgPartition, 0, width * rowsInPartition);
    memset(gradientXPartition, 0, width * rowsInPartition * sizeof(float));
    memset(gradientYPartition, 0, width * rowsInPartition * sizeof(float));
    memset(magnitudePartition, 0, width * rowsInPartition * sizeof(float));
    memset(directionPartition, 0, width * rowsInPartition * sizeof(float));

    // Setup CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("Rank %d: No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cudaSetDevice(rank % deviceCount);

    // Allocate device memory - only for our partition
    unsigned char *d_rgbData, *d_grayData, *d_blurredData, *d_edgesData;
    float *d_gradientX, *d_gradientY, *d_magnitude, *d_direction;

    cudaMalloc(&d_rgbData, width * height * 3); // Full RGB image
    cudaMalloc(&d_grayData, width * rowsInPartition);
    cudaMalloc(&d_blurredData, width * rowsInPartition);
    cudaMalloc(&d_edgesData, width * rowsInPartition);
    cudaMalloc(&d_gradientX, width * rowsInPartition * sizeof(float));
    cudaMalloc(&d_gradientY, width * rowsInPartition * sizeof(float));
    cudaMalloc(&d_magnitude, width * rowsInPartition * sizeof(float));
    cudaMalloc(&d_direction, width * rowsInPartition * sizeof(float));

    // Copy RGB data to device (full image needed for grayscale conversion)
    cudaMemcpy(d_rgbData, rgbImg.data, width * height * 3, cudaMemcpyHostToDevice);

    // Define kernel block dimensions
    dim3 blockDim(16, 16);

    // Set up grid dimensions for the partition
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (rowsInPartition + blockDim.y - 1) / blockDim.y);

    // Step 4: Convert RGB to grayscale using CUDA (only for this rank's partition)
    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_rgbData, d_grayData, width, height, startRow, endRow);
    cudaDeviceSynchronize();

    // Copy grayscale result back to host
    cudaMemcpy(grayImgPartition, d_grayData, width * rowsInPartition, cudaMemcpyDeviceToHost);

    // Step 5: Apply Gaussian blur using CUDA
    gaussianBlurKernel<<<gridDim, blockDim>>>(d_grayData, d_blurredData, width, height, startRow, endRow);
    cudaDeviceSynchronize();

    // Copy blurred result back to host
    cudaMemcpy(blurredImgPartition, d_blurredData, width * rowsInPartition, cudaMemcpyDeviceToHost);

    // Step 6: Apply Sobel operators for edge detection
    sobelXKernel<<<gridDim, blockDim>>>(d_blurredData, d_gradientX, width, height, startRow, endRow);
    sobelYKernel<<<gridDim, blockDim>>>(d_blurredData, d_gradientY, width, height, startRow, endRow);
    cudaDeviceSynchronize();

    // Step 7: Compute gradient magnitude and direction
    gradientKernel<<<gridDim, blockDim>>>(d_gradientX, d_gradientY, d_magnitude, d_direction, width, height, startRow, endRow);
    cudaDeviceSynchronize();

    // Step 8: Apply non-maximum suppression
    nonMaxSuppressionKernel<<<gridDim, blockDim>>>(d_magnitude, d_direction, d_edgesData, width, height, startRow, endRow);
    cudaDeviceSynchronize();

    // Copy edge detection result back to host
    cudaMemcpy(edgesImgPartition, d_edgesData, width * rowsInPartition, cudaMemcpyDeviceToHost);

    // Free CUDA memory
    cudaFree(d_rgbData);
    cudaFree(d_grayData);
    cudaFree(d_blurredData);
    cudaFree(d_edgesData);
    cudaFree(d_gradientX);
    cudaFree(d_gradientY);
    cudaFree(d_magnitude);
    cudaFree(d_direction);

    // Step 9: Use MPI I/O to write the output binary file in parallel
    MPI_File fh;
    MPI_Status status;

    // Open the file
    MPI_File_open(MPI_COMM_WORLD, outputBinary, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // Write header (only rank 0)
    if (rank == 0)
    {
        MPI_File_write(fh, &width, 1, MPI_INT, &status);
        MPI_File_write(fh, &height, 1, MPI_INT, &status);
    }

    // Wait for header to be written
    MPI_Barrier(MPI_COMM_WORLD);
    // ------ TIMING CODE -------
    /* <<<  end the timed region  >>> */
    MPI_Barrier(MPI_COMM_WORLD); // full sync so ranks stop together

    if (rank == 0)
    {
        uint64_t t1 = clock_now();
        uint64_t local_cycles = t1 - t0;
        printf("Rank %d: Total cycles: %lu\n", rank, local_cycles);
    }

    // Calculate offset for each rank's data
    MPI_Offset headerSize = 2 * sizeof(int);
    MPI_Offset offset = headerSize + startRow * width;

    // Write each rank's portion of the data (edge detection result)
    MPI_File_write_at(fh, offset, edgesImgPartition,
                      rowsInPartition * width, MPI_UNSIGNED_CHAR, &status);

    // Close the file
    MPI_File_close(&fh);

    // Step 10: Convert binary to PNG (only rank 0)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Rank %d: Converting binary to PNG %s\n", rank, outputPNG);
        GrayImage outputImg = binaryToGray(outputBinary);
        if (outputImg.width == 0 || outputImg.height == 0)
        {
            printf("Error reading output binary file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        writeGrayPNG(outputPNG, &outputImg);
        free(outputImg.data);
    }

    // Clean up
    free(grayImgPartition);
    free(blurredImgPartition);
    free(edgesImgPartition);
    free(gradientXPartition);
    free(gradientYPartition);
    free(magnitudePartition);
    free(directionPartition);
    free(rgbImg.data);

    MPI_Finalize();
    return 0;
}