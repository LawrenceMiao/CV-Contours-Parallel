#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Utility: Check CUDA errors (minimal version)
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ---------------------------------------------------------------------------
// Simple PNG reading using libpng (rank 0 only)
unsigned char* readPNG(const char* filename, int &width, int &height, int &channels)
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return nullptr;
    }

    // Initialize and check libpng structures
    png_structp png_ptr  = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop   info_ptr = png_create_info_struct(png_ptr);
    if(!png_ptr || !info_ptr) {
        fprintf(stderr, "png_create_read_struct or png_create_info_struct failed\n");
        fclose(fp);
        return nullptr;
    }

    if(setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during init_io\n");
        fclose(fp);
        return nullptr;
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    width    = png_get_image_width(png_ptr, info_ptr);
    height   = png_get_image_height(png_ptr, info_ptr);
    channels = png_get_channels(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    // Convert palette/gray images to RGB or RGBA as needed
    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if(png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if(bit_depth == 16)
        png_set_strip_16(png_ptr);
    if(color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
       color_type == PNG_COLOR_TYPE_GRAY)
        png_set_gray_to_rgb(png_ptr);

    // Update info after transformations
    png_read_update_info(png_ptr, info_ptr);

    width    = png_get_image_width(png_ptr, info_ptr);
    height   = png_get_image_height(png_ptr, info_ptr);
    channels = png_get_channels(png_ptr, info_ptr); // 3 or 4

    // Read row-by-row
    unsigned char* data = (unsigned char*)malloc(width * height * channels);
    png_bytep* row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)(data + y * width * channels);
    }
    png_read_image(png_ptr, row_pointers);

    free(row_pointers);
    fclose(fp);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    return data;
}

// ---------------------------------------------------------------------------
// CUDA Kernels (minimal examples)

// Convert RGB to grayscale
__global__ void kernel_rgb_to_grayscale(const unsigned char* d_in,
                                        unsigned char* d_out,
                                        int width, int height, int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    int index = (idy * width + idx) * channels;
    // Weighted average or simple average
    float r = d_in[index + 0];
    float g = d_in[index + 1];
    float b = d_in[index + 2];
    unsigned char gray = static_cast<unsigned char>(0.299f*r + 0.587f*g + 0.114f*b);

    d_out[idy * width + idx] = gray;
}

// Simple 2D convolution (Gaussian or Sobel) example
__global__ void kernel_convolution_2d(const unsigned char* d_in,
                                      unsigned char* d_out,
                                      const float* d_kernel,
                                      int kWidth, int kHeight,
                                      int width, int height)
{
    int halfW = kWidth  / 2;
    int halfH = kHeight / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    // Convolve with the kernel
    for(int ky = -halfH; ky <= halfH; ky++){
        for(int kx = -halfW; kx <= halfW; kx++){
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            float pixelVal = static_cast<float>(d_in[iy * width + ix]);
            float kernelVal = d_kernel[(ky+halfH)*kWidth + (kx+halfW)];
            sum += pixelVal * kernelVal;
        }
    }
    // Clamp to [0..255]
    sum = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
    d_out[y * width + x] = static_cast<unsigned char>(sum);
}

// Non-Maximum Suppression (very simplified, 8 directions)
__global__ void kernel_non_max_suppression(const unsigned char* d_mag,
                                           const float* d_gx,
                                           const float* d_gy,
                                           unsigned char* d_out,
                                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) {
        if (x < width && y < height) d_out[y * width + x] = 0;
        return;
    }

    // gradient direction
    float gx = d_gx[y * width + x];
    float gy = d_gy[y * width + x];
    float mag = d_mag[y * width + x];

    // angle in degrees or just use approximate direction
    float angle = atan2f(gy, gx) * 180.0f / 3.14159f;
    if (angle < 0) angle += 180.0f; // keep within [0, 180)

    unsigned char v1, v2;
    // Decide which neighbors to compare
    if ((angle > 0   && angle <= 22.5f) || (angle > 157.5f && angle <= 180.0f)) {
        v1 = d_mag[y * width + (x-1)];
        v2 = d_mag[y * width + (x+1)];
    } else if (angle > 22.5f && angle <= 67.5f) {
        v1 = d_mag[(y-1) * width + (x-1)];
        v2 = d_mag[(y+1) * width + (x+1)];
    } else if (angle > 67.5f && angle <= 112.5f) {
        v1 = d_mag[(y-1) * width + x];
        v2 = d_mag[(y+1) * width + x];
    } else { // 112.5f -> 157.5f
        v1 = d_mag[(y-1) * width + (x+1)];
        v2 = d_mag[(y+1) * width + (x-1)];
    }

    if (mag >= v1 && mag >= v2)
        d_out[y * width + x] = mag;
    else
        d_out[y * width + x] = 0;
}

// Thresholding -> binary
__global__ void kernel_threshold(const unsigned char* d_in,
                                 unsigned char* d_out,
                                 int width, int height,
                                 unsigned char thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char val = d_in[y * width + x];
    d_out[y * width + x] = (val >= thresh) ? 255 : 0;
}

// ---------------------------------------------------------------------------
// Main Program
int main(int argc, char* argv[])
{
    // MPI Init
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse any command-line arguments if needed
    // e.g. usage: mpirun -np <size> ./contour_mpi_cuda <input.png> [threshold=50]
    if(argc < 2 && rank == 0) {
        fprintf(stderr, "Usage: mpirun -np <size> ./contour_mpi_cuda images/input.png [threshold=50]\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    const char* inputFile = argv[1];
    unsigned char thresholdVal = 50; // default
    if(argc >= 3) {
        thresholdVal = static_cast<unsigned char>(atoi(argv[2]));
    }

    // Rank 0 reads image
    int width = 0, height = 0, channels = 0;
    unsigned char *hostImage = nullptr;
    if(rank == 0) {
        hostImage = readPNG(inputFile, width, height, channels);
        if(!hostImage) {
            fprintf(stderr, "Error reading PNG file!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        printf("Image loaded. W=%d, H=%d, C=%d\n", width, height, channels);
    }

    // Broadcast metadata (width, height, channels) so all ranks know
    MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // For simplicity, we broadcast the entire image to all ranks
    // (For memory scalability, you'd scatter a portion to each rank instead.)
    size_t imgSize = width * height * channels * sizeof(unsigned char);
    if(rank != 0) {
        hostImage = (unsigned char*)malloc(imgSize);
    }
    MPI_Bcast(hostImage, width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Allocate GPU memory
    unsigned char *d_in = nullptr, *d_gray = nullptr, *d_temp = nullptr;
    CUDA_CHECK( cudaMalloc(&d_in,   imgSize) );  // store input image
    // We'll store grayscale as 1 channel: width*height
    size_t graySize = width * height * sizeof(unsigned char);
    CUDA_CHECK( cudaMalloc(&d_gray, graySize) );
    CUDA_CHECK( cudaMalloc(&d_temp, graySize) );

    // Copy input image to GPU
    CUDA_CHECK( cudaMemcpy(d_in, hostImage, imgSize, cudaMemcpyHostToDevice) );

    // 1) Convert to Grayscale (only if channels == 3 or 4)
    if(channels == 3 || channels == 4) {
        dim3 block(16,16);
        dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
        kernel_rgb_to_grayscale<<<grid, block>>>(d_in, d_gray, width, height, channels);
        cudaDeviceSynchronize();
    } else {
        // If it's already grayscale, just copy into d_gray
        CUDA_CHECK( cudaMemcpy(d_gray, d_in, graySize, cudaMemcpyDeviceToDevice) );
    }

    // 2) Gaussian Smoothing
    // For demonstration, define a 3x3 kernel for small Gaussian
    float h_gauss[9] = { 1/16.f, 2/16.f, 1/16.f,
                         2/16.f, 4/16.f, 2/16.f,
                         1/16.f, 2/16.f, 1/16.f };
    float *d_gauss = nullptr;
    CUDA_CHECK( cudaMalloc(&d_gauss, 9*sizeof(float)) );
    CUDA_CHECK( cudaMemcpy(d_gauss, h_gauss, 9*sizeof(float), cudaMemcpyHostToDevice) );

    {
        dim3 block(16,16);
        dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
        kernel_convolution_2d<<<grid, block>>>(d_gray, d_temp, d_gauss,
                                               3, 3, width, height);
        cudaDeviceSynchronize();
    }

    // d_temp now holds the smoothed image.  Copy back to d_gray for subsequent steps
    CUDA_CHECK( cudaMemcpy(d_gray, d_temp, graySize, cudaMemcpyDeviceToDevice) );
    CUDA_CHECK( cudaFree(d_gauss) );

    // 3) Edge Detection (Sobel)
    // We'll do two separate convolutions: Gx, Gy
    // Typically, Gx kernel = [-1, 0, 1; -2, 0, 2; -1, 0, 1], etc.
    float h_sobelx[9] = { -1.f,  0.f,  1.f,
                          -2.f,  0.f,  2.f,
                          -1.f,  0.f,  1.f };
    float h_sobely[9] = {  1.f,  2.f,  1.f,
                           0.f,  0.f,  0.f,
                          -1.f, -2.f, -1.f };

    float *d_sobelx = nullptr, *d_sobely = nullptr;
    CUDA_CHECK( cudaMalloc(&d_sobelx, 9*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_sobely, 9*sizeof(float)) );
    CUDA_CHECK( cudaMemcpy(d_sobelx, h_sobelx, 9*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_sobely, h_sobely, 9*sizeof(float), cudaMemcpyHostToDevice) );

    // We'll store Gx, Gy as float arrays
    float *d_gx = nullptr, *d_gy = nullptr;
    CUDA_CHECK( cudaMalloc(&d_gx, width*height*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_gy, width*height*sizeof(float)) );

    // Use kernel_convolution_2d but to floats for Gx, Gy
    // For brevity, define a separate kernel that writes floats:
    // (Here, to keep code short, we’ll do a quick inline approach.)
    auto kernel_convolution_2d_float = [] __global__ (const unsigned char* in, float* out,
                                                      const float* ker, int kw, int kh,
                                                      int w, int h) {
        int halfW = kw/2, halfH = kh/2;
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        if(x>=w || y>=h) return;

        float sum=0.f;
        for(int ky=-halfH; ky<=halfH; ky++){
            for(int kx=-halfW; kx<=halfW; kx++){
                int ix = min(max(x+kx,0), w-1);
                int iy = min(max(y+ky,0), h-1);
                float pix = (float)in[iy*w + ix];
                float kVal = ker[(ky+halfH)*kw + (kx+halfW)];
                sum += pix*kVal;
            }
        }
        out[y*w + x] = sum;
    };

    // Launch for Gx
    dim3 block(16,16), grid((width+15)/16, (height+15)/16);
    kernel_convolution_2d_float<<<grid, block>>>(d_gray, d_gx, d_sobelx, 3,3, width, height);
    // Launch for Gy
    kernel_convolution_2d_float<<<grid, block>>>(d_gray, d_gy, d_sobely, 3,3, width, height);
    cudaDeviceSynchronize();

    // Now compute magnitude from Gx, Gy into d_temp (unsigned char)
    // We’ll do sqrt(Gx^2 + Gy^2). For brevity, define a small inline kernel:
    auto kernel_magnitude = [] __global__ (const float* gx, const float* gy,
                                           unsigned char* out, int w, int h) {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        if(x>=w || y>=h) return;
        float fx = gx[y*w+x];
        float fy = gy[y*w+x];
        float mag = sqrtf(fx*fx + fy*fy);
        if(mag>255.f) mag=255.f;
        out[y*w + x] = (unsigned char)(mag);
    };
    kernel_magnitude<<<grid, block>>>(d_gx, d_gy, d_temp, width, height);
    cudaDeviceSynchronize();

    // 4) Non-Maximum Suppression
    // We'll produce a new buffer d_nms
    unsigned char* d_nms = nullptr;
    CUDA_CHECK( cudaMalloc(&d_nms, graySize) );
    kernel_non_max_suppression<<<grid, block>>>(d_temp, d_gx, d_gy, d_nms, width, height);
    cudaDeviceSynchronize();

    // 5) Threshold -> binary
    unsigned char* d_binary = nullptr;
    CUDA_CHECK( cudaMalloc(&d_binary, graySize) );
    kernel_threshold<<<grid, block>>>(d_nms, d_binary, width, height, thresholdVal);
    cudaDeviceSynchronize();

    // (Optional) Copy back to host to gather / output results
    unsigned char* hostResult = (unsigned char*)malloc(graySize);
    CUDA_CHECK( cudaMemcpy(hostResult, d_binary, graySize, cudaMemcpyDeviceToHost) );

    // In a multi-rank scenario, you might gather partial results from each rank.
    // Here we did a naive broadcast of the entire image to each rank, so each
    // rank has a complete result. For demonstration, let's just have rank 0
    // print out a small text message or write a final result.

    if(rank == 0) {
        printf("Edge detection pipeline complete.\n");
        // You can write 'hostResult' as a grayscale PNG if desired,
        // or store it for further processing. That code is omitted for brevity.
    }

    // Clean up
    free(hostImage);
    free(hostResult);
    CUDA_CHECK( cudaFree(d_in) );
    CUDA_CHECK( cudaFree(d_gray) );
    CUDA_CHECK( cudaFree(d_temp) );
    CUDA_CHECK( cudaFree(d_nms) );
    CUDA_CHECK( cudaFree(d_binary) );
    CUDA_CHECK( cudaFree(d_gx) );
    CUDA_CHECK( cudaFree(d_gy) );
    CUDA_CHECK( cudaFree(d_sobelx) );
    CUDA_CHECK( cudaFree(d_sobely) );

    MPI_Finalize();
    return 0;
}
