#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>

// Structure to hold image data
typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

// ------------------ CUDA Kernels ------------------
// RGB->Grayscale (NCHW layout)
__global__ void grayscale_cuda(const float *rgb, float *gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    int plane = width * height;
    gray[idx] = 0.299f * rgb[0 * plane + idx] + 0.587f * rgb[1 * plane + idx] + 0.114f * rgb[2 * plane + idx];
}

// Gaussian smoothing kernel
__global__ void gaussian_smooth_cuda(const float *in, float *out,
                                     const float *kernel, int ksize,
                                     int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int half = ksize / 2;
    float sum = 0.0f, ks = 0.0f;
    for (int dy = -half; dy <= half; dy++)
    {
        for (int dx = -half; dx <= half; dx++)
        {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);
            float val = in[iy * width + ix];
            float kval = kernel[(dy + half) * ksize + (dx + half)];
            sum += val * kval;
            ks += kval;
        }
    }
    out[y * width + x] = (ks > 0 ? sum / ks : sum);
}

// Sobel kernel for computing gradients
__global__ void sobel_cuda(const float *in, // grayscale input
                           float *gx,       // horizontal gradient
                           float *gy,       // vertical gradient
                           int width,
                           int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // 3×3 Sobel coefficients
    const float kx[3][3] = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};

    const float ky[3][3] = {{1, 2, 1},
                            {0, 0, 0},
                            {-1, -2, -1}};

    float sx = 0.f, sy = 0.f;

    // Convolve in a 3×3 window (with clamped borders)
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);
            float p = in[iy * width + ix];
            sx += p * kx[dy + 1][dx + 1];
            sy += p * ky[dy + 1][dx + 1];
        }

    int idx = y * width + x;
    gx[idx] = sx;
    gy[idx] = sy;
}

// Kernel to compute magnitude
__global__ void mag_cuda(const float *gx, const float *gy,
                         float *mag, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float gxv = gx[idx], gyv = gy[idx];
    mag[idx] = sqrtf(gxv * gxv + gyv * gyv);
}

// Non-maximum Suppression Kernel
__global__ void nms_cuda(const float *mag,
                         const float *gx,
                         const float *gy,
                         unsigned char *nms,
                         int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * width + x;
    float g = mag[idx];

    // Angle in degrees [0,180)
    float angle = atan2f(gy[idx], gx[idx]) * 57.2957795f; // 180/pi
    if (angle < 0)
        angle += 180.f;

    float g1, g2;
    if ((angle <= 22.5f) || (angle > 157.5f))
    { // 0°
        g1 = mag[idx - 1];
        g2 = mag[idx + 1];
    }
    else if (angle <= 67.5f)
    { // 45°
        g1 = mag[(y - 1) * width + (x - 1)];
        g2 = mag[(y + 1) * width + (x + 1)];
    }
    else if (angle <= 112.5f)
    { // 90°
        g1 = mag[(y - 1) * width + x];
        g2 = mag[(y + 1) * width + x];
    }
    else
    { // 135°
        g1 = mag[(y - 1) * width + (x + 1)];
        g2 = mag[(y + 1) * width + (x - 1)];
    }

    nms[idx] = (g >= g1 && g >= g2) ? (unsigned char)__float2int_rn(fminf(g, 255.f))
                                    : 0;
}

__global__ void thresh_cuda(const unsigned char *in, // NMS result
                            unsigned char *out,      // binary mask
                            int width,
                            int height,
                            unsigned char thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    unsigned char v = in[idx];
    out[idx] = (v >= thresh) ? 255 : 0;
}

// ------------------ Host Wrappers ------------------
void create_gaussian_kernel(float *kernel, int ksize, float sigma)
{
    int half = ksize / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++)
    {
        for (int x = -half; x <= half; x++)
        {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(y + half) * ksize + (x + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < ksize * ksize; i++)
        kernel[i] /= sum;
}

void apply_gaussian_smooth(float *d_in, float *d_out, int width, int height,
                           int ksize, float sigma)
{
    float *h_kernel = (float *)malloc(ksize * ksize * sizeof(float));
    create_gaussian_kernel(h_kernel, ksize, sigma);
    float *d_kernel;
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
    gaussian_smooth_cuda<<<grid, block>>>(d_in, d_out, d_kernel, ksize, width, height);
    cudaDeviceSynchronize();

    cudaFree(d_kernel);
    free(h_kernel);
}

// ------------------ PNG Helpers ------------------
Image *readPNG(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fclose(fp);
        return NULL;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int channels = png_get_channels(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA || color_type == PNG_COLOR_TYPE_GRAY)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);
    channels = png_get_channels(png_ptr, info_ptr);

    size_t rowbytes = width * channels * sizeof(unsigned char);
    unsigned char *data = (unsigned char *)malloc(height * rowbytes);
    if (!data)
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_bytep *row_ptrs = (png_bytep *)malloc(height * sizeof(png_bytep));
    if (!row_ptrs)
    {
        free(data);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    for (int y = 0; y < height; y++)
        row_ptrs[y] = data + y * rowbytes;

    png_read_image(png_ptr, row_ptrs);

    free(row_ptrs);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    Image *img = (Image *)malloc(sizeof(Image));
    img->data = data;
    img->width = width;
    img->height = height;
    img->channels = channels;

    return img;
}

int writePNG(const char *filename, unsigned char *data, int width, int height, int channels)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error opening %s for writing\n", filename);
        return -1;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fclose(fp);
        return -1;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return -1;
    }
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return -1;
    }

    png_init_io(png_ptr, fp);
    int color_type = (channels == 1 ? PNG_COLOR_TYPE_GRAY : (channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA));
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    size_t rowbytes = width * channels * sizeof(unsigned char);
    png_bytep *row_ptrs = (png_bytep *)malloc(height * sizeof(png_bytep));
    for (int y = 0; y < height; y++)
        row_ptrs[y] = data + y * rowbytes;

    png_write_image(png_ptr, row_ptrs);
    png_write_end(png_ptr, info_ptr);

    free(row_ptrs);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return 0;
}

// ------------------ MPI I/O Functions ------------------
void write_raw_image_mpi(MPI_File *fh, unsigned char *data, int width, int local_height,
                         MPI_Offset offset, MPI_Comm comm)
{
    MPI_Status status;
    MPI_File_write_at_all(*fh, offset, data, width * local_height, MPI_UNSIGNED_CHAR, &status);
}

void read_raw_image_mpi(MPI_File *fh, unsigned char *data, int width, int local_height,
                        MPI_Offset offset, MPI_Comm comm)
{
    MPI_Status status;
    MPI_File_read_at_all(*fh, offset, data, width * local_height, MPI_UNSIGNED_CHAR, &status);
}

// ------------------ Main Pipeline ------------------
int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc < 3)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s input.png output.png\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Read input image (only rank 0 reads the full PNG)
    Image *img = NULL;
    int width = 0, height = 0, channels = 0;

    if (rank == 0)
    {
        img = readPNG(argv[1]);
        if (!img)
        {
            fprintf(stderr, "Error reading input image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        width = img->width;
        height = img->height;
        channels = img->channels;
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Convert PNG to raw binary data and distribute with MPI I/O
    const char *temp_raw_file = "temp_input.raw";

    // Compute split offsets for horizontal strips
    int base = height / size, rem = height % size;
    int *counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        int h = base + (i < rem ? 1 : 0);
        counts[i] = h * width * channels;
        displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }

    // Get my chunk's height
    int my_height = base + (rank < rem ? 1 : 0);
    int my_count = my_height * width * channels;

    // Create temporary raw file (rank 0 only)
    if (rank == 0)
    {
        FILE *fp = fopen(temp_raw_file, "wb");
        if (!fp)
        {
            fprintf(stderr, "Error creating temporary raw file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        fwrite(img->data, 1, height * width * channels, fp);
        fclose(fp);
    }

    // Ensure all processes see the file
    MPI_Barrier(MPI_COMM_WORLD);

    // Open raw file with MPI I/O
    MPI_File fh_in;
    MPI_File_open(MPI_COMM_WORLD, temp_raw_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);

    // Each process reads its own chunk
    unsigned char *chunk = (unsigned char *)malloc(my_count);
    MPI_Offset offset = displs[rank];
    read_raw_image_mpi(&fh_in, chunk, width * channels, my_height, offset, MPI_COMM_WORLD);

    // Close input file
    MPI_File_close(&fh_in);

    // Convert chunk to NCHW float format for CUDA processing
    size_t rgbSize = 3 * width * my_height * sizeof(float);
    size_t graySize = width * my_height * sizeof(float);
    float *rgbf = (float *)malloc(rgbSize);

    for (int c = 0; c < 3; c++)
        for (int y = 0; y < my_height; y++)
            for (int x = 0; x < width; x++)
            {
                int pid = (y * width + x) * channels + c;
                rgbf[c * width * my_height + y * width + x] = (float)chunk[pid];
            }

    // Launch on GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
    {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    cudaSetDevice(rank % device_count);

    float *d_rgb, *d_gray, *d_smooth;
    cudaMalloc(&d_rgb, rgbSize);
    cudaMalloc(&d_gray, graySize);
    cudaMalloc(&d_smooth, graySize);
    cudaMemcpy(d_rgb, rgbf, rgbSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((width + 15) / 16, (my_height + 15) / 16);
    grayscale_cuda<<<grid, block>>>(d_rgb, d_gray, width, my_height);
    cudaDeviceSynchronize();

    apply_gaussian_smooth(d_gray, d_smooth, width, my_height, 5, 1.0f);

    // ---------- Sobel gradients ----------
    float *d_gx, *d_gy;
    cudaMalloc(&d_gx, graySize);
    cudaMalloc(&d_gy, graySize);

    sobel_cuda<<<grid, block>>>(d_smooth, d_gx, d_gy, width, my_height);
    cudaDeviceSynchronize();

    // ---------- Magnitude and Non-max suppression ----------
    float *d_mag;
    cudaMalloc(&d_mag, graySize);
    unsigned char *d_nms;
    cudaMalloc(&d_nms, graySize); // uchar output

    mag_cuda<<<grid, block>>>(d_gx, d_gy, d_mag, width, my_height);
    nms_cuda<<<grid, block>>>(d_mag, d_gx, d_gy, d_nms, width, my_height);
    cudaDeviceSynchronize();

    // ---------- Threshold to binary ----------
    unsigned char *d_bin;
    cudaMalloc(&d_bin, graySize); // graySize bytes

    const unsigned char TH = 75; // Threshold
    thresh_cuda<<<grid, block>>>(d_nms, d_bin, width, my_height, TH);
    cudaDeviceSynchronize();

    // Allocate host buffer for result
    unsigned char *outc = (unsigned char *)malloc(width * my_height);
    cudaMemcpy(outc, d_bin, width * my_height, cudaMemcpyDeviceToHost);

    // Use MPI I/O to write results (each process writes its part)
    const char *temp_out_file = "temp_output.raw";
    MPI_File fh_out;

    // Calculate output offsets (each process writes one row at a time)
    int *out_displs = (int *)malloc(size * sizeof(int));
    int *out_counts = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        int h = base + (i < rem ? 1 : 0);
        out_counts[i] = h * width;
        out_displs[i] = (i == 0) ? 0 : out_displs[i - 1] + out_counts[i - 1];
    }

    // Open output file collectively
    MPI_File_open(MPI_COMM_WORLD, temp_out_file,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh_out);

    // Set file size (only root)
    if (rank == 0)
    {
        MPI_File_set_size(fh_out, width * height);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Write data
    MPI_Offset out_offset = out_displs[rank];
    write_raw_image_mpi(&fh_out, outc, width, my_height, out_offset, MPI_COMM_WORLD);

    // Close output file
    MPI_File_close(&fh_out);

    // Process gradient images - Gx and Gy
    // Allocate buffers for gradient results
    unsigned char *gx_uc = (unsigned char *)malloc(width * my_height);
    unsigned char *gy_uc = (unsigned char *)malloc(width * my_height);

    // Copy from device and normalize
    float *gx_host = (float *)malloc(width * my_height * sizeof(float));
    float *gy_host = (float *)malloc(width * my_height * sizeof(float));
    cudaMemcpy(gx_host, d_gx, width * my_height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gy_host, d_gy, width * my_height * sizeof(float), cudaMemcpyDeviceToHost);

    // Find local maximum gradient values
    float local_max_gx = 0.0f, local_max_gy = 0.0f;
    for (int i = 0; i < width * my_height; i++)
    {
        local_max_gx = fmaxf(local_max_gx, fabsf(gx_host[i]));
        local_max_gy = fmaxf(local_max_gy, fabsf(gy_host[i]));
    }

    // Reduce to global max across all ranks
    float max_gx, max_gy;
    MPI_Allreduce(&local_max_gx, &max_gx, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_gy, &max_gy, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    // Scale factors
    float scale_gx = (max_gx > 0.0f) ? 255.0f / max_gx : 1.0f;
    float scale_gy = (max_gy > 0.0f) ? 255.0f / max_gy : 1.0f;

    // Scale and convert to unsigned char
    for (int i = 0; i < width * my_height; i++)
    {
        int v1 = (int)fminf(255.0f, fabsf(gx_host[i]) * scale_gx);
        int v2 = (int)fminf(255.0f, fabsf(gy_host[i]) * scale_gy);
        gx_uc[i] = (unsigned char)v1;
        gy_uc[i] = (unsigned char)v2;
    }

    // Write gradient images using MPI I/O
    // Write Gx gradient
    const char *temp_gx_file = "temp_gx.raw";
    MPI_File fh_gx;
    MPI_File_open(MPI_COMM_WORLD, temp_gx_file,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh_gx);
    if (rank == 0)
    {
        MPI_File_set_size(fh_gx, width * height);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    write_raw_image_mpi(&fh_gx, gx_uc, width, my_height, out_offset, MPI_COMM_WORLD);
    MPI_File_close(&fh_gx);

    // Write Gy gradient
    const char *temp_gy_file = "temp_gy.raw";
    MPI_File fh_gy;
    MPI_File_open(MPI_COMM_WORLD, temp_gy_file,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh_gy);
    if (rank == 0)
    {
        MPI_File_set_size(fh_gy, width * height);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    write_raw_image_mpi(&fh_gy, gy_uc, width, my_height, out_offset, MPI_COMM_WORLD);
    MPI_File_close(&fh_gy);

    // Convert raw output to PNG (only rank 0)
    if (rank == 0)
    {
        // Read the complete raw output data
        FILE *fp = fopen(temp_out_file, "rb");
        if (!fp)
        {
            fprintf(stderr, "Error opening temporary output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        unsigned char *full_output = (unsigned char *)malloc(width * height);
        fread(full_output, 1, width * height, fp);
        fclose(fp);

        // Write as PNG
        writePNG(argv[2], full_output, width, height, 1);

        // Read and convert Gx
        fp = fopen(temp_gx_file, "rb");
        if (fp)
        {
            unsigned char *full_gx = (unsigned char *)malloc(width * height);
            fread(full_gx, 1, width * height, fp);
            fclose(fp);
            writePNG("gx.png", full_gx, width, height, 1);
            free(full_gx);
        }

        // Read and convert Gy
        fp = fopen(temp_gy_file, "rb");
        if (fp)
        {
            unsigned char *full_gy = (unsigned char *)malloc(width * height);
            fread(full_gy, 1, width * height, fp);
            fclose(fp);
            writePNG("gy.png", full_gy, width, height, 1);
            free(full_gy);
        }

        // Clean up temporary files
        remove(temp_raw_file);
        remove(temp_out_file);
        remove(temp_gx_file);
        remove(temp_gy_file);

        free(full_output);
        if (img != NULL)
        {
            free(img->data);
            free(img);
        }
    }

    // Cleanup
    free(chunk);
    free(rgbf);
    free(outc);
    free(counts);
    free(displs);
    free(out_counts);
    free(out_displs);
    free(gx_host);
    free(gy_host);
    free(gx_uc);
    free(gy_uc);

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_smooth);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_mag);
    cudaFree(d_nms);
    cudaFree(d_bin);

    MPI_Finalize();
    return 0;
}