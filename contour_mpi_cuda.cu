#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------ libpng I/O ------------------
unsigned char* readPNG(const char* filename, int* width, int* height, int* channels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return NULL;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);
    *width    = png_get_image_width(png_ptr, info_ptr);
    *height   = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);

    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);
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
    *channels = png_get_channels(png_ptr, info_ptr);

    size_t rowbytes = (*width) * (*channels) * sizeof(unsigned char);
    unsigned char* data = (unsigned char*)malloc((*height) * rowbytes);
    if (!data) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_bytep* row_ptrs = (png_bytep*)malloc((*height) * sizeof(png_bytep));
    if (!row_ptrs) {
        free(data);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    for (int y = 0; y < *height; y++)
        row_ptrs[y] = data + y * rowbytes;

    png_read_image(png_ptr, row_ptrs);

    free(row_ptrs);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return data;
}

// ------------------ libpng Output ------------------
int writePNG(const char* filename, unsigned char* data, int width, int height, int channels) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening %s for writing\n", filename);
        return -1;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return -1;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return -1;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
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
    png_bytep* row_ptrs = (png_bytep*)malloc(height * sizeof(png_bytep));
    for (int y = 0; y < height; y++)
        row_ptrs[y] = data + y * rowbytes;

    png_write_image(png_ptr, row_ptrs);
    png_write_end(png_ptr, info_ptr);

    free(row_ptrs);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return 0;
}

// ------------------ CUDA Kernels ------------------
// RGB->Grayscale (NCHW layout)
__global__ void grayscale_cuda(const float *rgb, float *gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int plane = width * height;
    gray[idx] = 0.299f * rgb[0*plane + idx]
               + 0.587f * rgb[1*plane + idx]
               + 0.114f * rgb[2*plane + idx];
}

// Gaussian smoothing kernel
__global__ void gaussian_smooth_cuda(const float *in, float *out,
                                     const float *kernel, int ksize,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float sum = 0.0f, ks = 0.0f;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);
            float val = in[iy*width + ix];
            float kval = kernel[(dy+half)*ksize + (dx+half)];
            sum += val * kval;
            ks  += kval;
        }
    }
    out[y*width + x] = (ks > 0 ? sum/ks : sum);
}

// Sobel kernel for computing gradients
__global__ void sobel_cuda(const float *in,   // grayscale input
    float *gx,         // horizontal gradient
    float *gy,         // vertical   gradient
    int   width,
    int   height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= width || y >= height) return;

// 3×3 Sobel coefficients
const float kx[3][3] = { {-1, 0, 1},
      {-2, 0, 2},
      {-1, 0, 1} };

const float ky[3][3] = { { 1,  2,  1},
      { 0,  0,  0},
      {-1, -2, -1} };

float sx = 0.f, sy = 0.f;

// Convolve in a 3×3 window (with clamped borders)
for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
        int ix = min(max(x + dx, 0), width  - 1);
        int iy = min(max(y + dy, 0), height - 1);
        float p  = in[iy * width + ix];
        sx      += p * kx[dy + 1][dx + 1];
        sy      += p * ky[dy + 1][dx + 1];
    }

    int idx   = y * width + x;
    gx[idx] = sx;
    gy[idx] = sy;
}

// Kernel to compute magnitude
__global__ void mag_cuda(const float *gx, const float *gy,
    float *mag, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float gxv = gx[idx], gyv = gy[idx];
    mag[idx] = sqrtf(gxv * gxv + gyv * gyv);
}

// Non-maximum Surpression Kernel

__global__ void nms_cuda(const float *mag,
    const float *gx,
    const float *gy,
    unsigned char *nms,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float g  = mag[idx];

    // Angle in degrees [0,180)
    float angle = atan2f(gy[idx], gx[idx]) * 57.2957795f;  // 180/pi
    if (angle < 0) angle += 180.f;

    float g1, g2;
    if ((angle <= 22.5f) || (angle > 157.5f)) {          // 0°
    g1 = mag[idx - 1];
    g2 = mag[idx + 1];
    } else if (angle <= 67.5f) {                         // 45°
    g1 = mag[(y - 1) * width + (x - 1)];
    g2 = mag[(y + 1) * width + (x + 1)];
    } else if (angle <= 112.5f) {                        // 90°
    g1 = mag[(y - 1) * width +  x    ];
    g2 = mag[(y + 1) * width +  x    ];
    } else {                                             // 135°
    g1 = mag[(y - 1) * width + (x + 1)];
    g2 = mag[(y + 1) * width + (x - 1)];
    }

    nms[idx] = (g >= g1 && g >= g2) ? (unsigned char)__float2int_rn(fminf(g,255.f))
                : 0;
}

__global__ void thresh_cuda(const unsigned char *in,  // NMS result
    unsigned char       *out, // binary mask
    int                  width,
    int                  height,
    unsigned char        thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char v = in[idx];
    out[idx] = (v >= thresh) ? 255 : 0;
}

// ------------------ Host Wrappers ------------------
void create_gaussian_kernel(float *kernel, int ksize, float sigma) {
    int half = ksize / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = expf(-(x*x + y*y)/(2.0f*sigma*sigma));
            kernel[(y+half)*ksize + (x+half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < ksize*ksize; i++)
        kernel[i] /= sum;
}

void apply_gaussian_smooth(float *d_in, float *d_out, int width, int height,
                           int ksize, float sigma) {
    float *h_kernel = (float*)malloc(ksize*ksize*sizeof(float));
    create_gaussian_kernel(h_kernel, ksize, sigma);
    float *d_kernel;
    cudaMalloc(&d_kernel, ksize*ksize*sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16), grid((width+15)/16,(height+15)/16);
    gaussian_smooth_cuda<<<grid,block>>>(d_in, d_out, d_kernel, ksize, width, height);
    cudaDeviceSynchronize();

    cudaFree(d_kernel);
    free(h_kernel);
}

// ------------------ Main Pipeline ------------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank==0) fprintf(stderr, "Error: This program requires 4 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int width=0, height=0, channels=0;
    unsigned char *img = NULL;
    if (rank == 0) {
        img = readPNG(argv[1], &width, &height, &channels);
        if (!img) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute split offsets
    int base = height/size, rem = height%size;
    int *counts = (int*)malloc(size*sizeof(int));
    int *displs = (int*)malloc(size*sizeof(int));
    for (int i=0; i<size; i++) {
        int h = base + (i<rem);
        counts[i] = h * width * channels;
        displs[i] = (i==0?0:displs[i-1] + counts[i-1]);
    }

    int myCount = counts[rank];
    unsigned char *chunk = (unsigned char*)malloc(myCount);
    MPI_Scatterv(img, counts, displs, MPI_UNSIGNED_CHAR,
                 chunk, myCount, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Convert chunk to NCHW float
    int myHeight = counts[rank]/(width*channels);
    size_t rgbSize  = 3 * width * myHeight * sizeof(float);
    size_t graySize =     width * myHeight * sizeof(float);
    float *rgbf = (float*)malloc(rgbSize);
    for(int c=0; c<3; c++)
    for(int y=0; y<myHeight; y++)
    for(int x=0; x<width; x++) {
        int pid = (y*width + x)*channels + c;
        rgbf[c*width*myHeight + y*width + x] = (float)chunk[pid];
    }

    // Launch on GPU
    cudaSetDevice(rank);
    float *d_rgb, *d_gray, *d_smooth;
    cudaMalloc(&d_rgb,   rgbSize);
    cudaMalloc(&d_gray,  graySize);
    cudaMalloc(&d_smooth,graySize);
    cudaMemcpy(d_rgb, rgbf, rgbSize, cudaMemcpyHostToDevice);

    dim3 block(16,16), grid((width+15)/16,(myHeight+15)/16);
    grayscale_cuda<<<grid,block>>>(d_rgb, d_gray, width, myHeight);
    cudaDeviceSynchronize();

    apply_gaussian_smooth(d_gray, d_smooth, width, myHeight, 5, 1.0f);

    // ---------- Sobel gradients ----------
    float *d_gx, *d_gy;
    cudaMalloc(&d_gx, graySize);
    cudaMalloc(&d_gy, graySize);

    sobel_cuda<<<grid, block>>>(d_smooth, d_gx, d_gy, width, myHeight);
    cudaDeviceSynchronize();

    // ---------- Magnitude and Non-max surpression ----------
    float        *d_mag;  cudaMalloc(&d_mag, graySize);
    unsigned char*d_nms;  cudaMalloc(&d_nms, graySize);   // uchar output

    mag_cuda <<<grid, block>>>(d_gx, d_gy, d_mag, width, myHeight);
    nms_cuda <<<grid, block>>>(d_mag, d_gx, d_gy, d_nms, width, myHeight);
    cudaDeviceSynchronize();

    // ---------- Threshold to binary ----------
    unsigned char *d_bin;  cudaMalloc(&d_bin, graySize);   // graySize bytes

    const unsigned char TH = 20;  // IMPORTANT: Static threshold
    thresh_cuda<<<grid, block>>>(d_nms, d_bin, width, myHeight, TH);
    cudaDeviceSynchronize();

    // cudaMemcpy(rgbf, d_smooth, graySize, cudaMemcpyDeviceToHost); // to output grayscale
    cudaMemcpy(rgbf, d_bin, graySize, cudaMemcpyDeviceToHost);

    // Convert to uchar
    unsigned char *outc = (unsigned char*)malloc(graySize);
    for (int i=0; i<width*myHeight; i++) {
        float v = rgbf[i];
        outc[i] = (unsigned char)(v<0?0:(v>255?255:v));
    }

            // Gather single-channel results back to rank 0
    unsigned char *full = NULL;
    if (rank == 0) {
        full = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    }

    int *recvCounts = (int*)malloc(size * sizeof(int));
    int *recvDispls = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int h_i = counts[i] / (width * channels);
        recvCounts[i] = h_i * width;
        int offsetRows = displs[i] / (width * channels);
        recvDispls[i] = offsetRows * width;
    }
    int sendCount = myHeight * width;

    MPI_Gatherv(outc, sendCount, MPI_UNSIGNED_CHAR,
                full, recvCounts, recvDispls,
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    /////// TEST OUTPUT - THIS OUTPUTS IMAGE FROM THE SOBEL GRADIENTS

    // --- Copy Sobel grads back to host and normalize to [0,255] ---
    int   pixCount = width * myHeight;          // number of pixels in this stripe
    float* gx_host = (float*)malloc(pixCount * sizeof(float));
    float* gy_host = (float*)malloc(pixCount * sizeof(float));
    cudaMemcpy(gx_host, d_gx, pixCount*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gy_host, d_gy, pixCount*sizeof(float), cudaMemcpyDeviceToHost);

    // Find local maximum absolute gradient
    float local_max_gx = 0.f, local_max_gy = 0.f;
    for(int i = 0; i < pixCount; i++){
        local_max_gx = fmaxf(local_max_gx, fabsf(gx_host[i]));
        local_max_gy = fmaxf(local_max_gy, fabsf(gy_host[i]));
    }

    // Reduce to global max across all ranks
    float max_gx, max_gy;
    MPI_Allreduce(&local_max_gx, &max_gx, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_gy, &max_gy, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    // Scale factors
    float scale_gx = (max_gx > 0.f ? 255.f / max_gx : 1.f);
    float scale_gy = (max_gy > 0.f ? 255.f / max_gy : 1.f);

    // Convert to unsigned char
    unsigned char *gx_uc = (unsigned char*)malloc(pixCount);
    unsigned char *gy_uc = (unsigned char*)malloc(pixCount);
    for(int i = 0; i < pixCount; i++){
        int v1 = (int)fminf(255.f, fabsf(gx_host[i]) * scale_gx);
        int v2 = (int)fminf(255.f, fabsf(gy_host[i]) * scale_gy);
        gx_uc[i] = (unsigned char)v1;
        gy_uc[i] = (unsigned char)v2;
    }

    free(gx_host);
    free(gy_host);

    // --- Gather gradient images back to rank 0 ---
    unsigned char *full_gx = NULL, *full_gy = NULL;
    if(rank == 0) {
        full_gx = (unsigned char*)malloc(width * height);
        full_gy = (unsigned char*)malloc(width * height);
    }

    // recvCounts / recvDispls are exactly as you set for your binary mask gather:
    //   recvCounts[i] = (counts[i]/(width*channels)) * width;
    //   recvDispls[i] = (displs[i]/(width*channels)) * width;
    MPI_Gatherv(gx_uc,       pixCount, MPI_UNSIGNED_CHAR,
                full_gx, recvCounts, recvDispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(gy_uc,       pixCount, MPI_UNSIGNED_CHAR,
                full_gy, recvCounts, recvDispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    free(gx_uc);
    free(gy_uc);

    // --- On rank 0, write out the two gradient PNGs ---
    if(rank == 0) {
        writePNG("gx.png", full_gx, width, height, 1);
        writePNG("gy.png", full_gy, width, height, 1);
        free(full_gx);
        free(full_gy);
    }
    
    /////// END TEST OUTPUT OF SOBEL GRADIENT (Could delete this later)

        // ----------------------------------------------------
    // DEBUG: write out raw NMS map  ->  nms.png
    // ----------------------------------------------------
    {
        /* 1. copy my stripe to host */
        int pixCount = myHeight * width;      // pixels in this rank
        unsigned char *nms_uc = (unsigned char*)malloc(pixCount);
        cudaMemcpy(nms_uc, d_nms, pixCount, cudaMemcpyDeviceToHost);

        /* 2. gather to rank‑0 (reuse recvCounts / recvDispls) */
        unsigned char *full_nms = NULL;
        if (rank == 0)
            full_nms = (unsigned char*)malloc(width * height);

        MPI_Gatherv(nms_uc,      pixCount, MPI_UNSIGNED_CHAR,
                    full_nms, recvCounts, recvDispls, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);

        free(nms_uc);

        /* 3. rank‑0 writes PNG */
        if (rank == 0) {
            writePNG("nms.png", full_nms, width, height, 1);
            free(full_nms);
        }
    }
    // ----------------------------------------------------

    // ----------------------------------------------------
    // DEBUG: write out thresholded NMS -> thresh_debug.png
    // ----------------------------------------------------
    {
        // 1. Launch with a low test threshold (e.g. 20)
        const unsigned char DEBUG_T = 20;
        thresh_cuda<<<grid,block>>>(d_nms, d_bin, width, myHeight, DEBUG_T);
        cudaDeviceSynchronize();

        // 2. Copy back stripe
        int pixCount = myHeight * width;
        unsigned char *th_uc = (unsigned char*)malloc(pixCount);
        cudaMemcpy(th_uc, d_bin, pixCount, cudaMemcpyDeviceToHost);

        // 3. Gather to rank 0
        unsigned char *full_th = NULL;
        if(rank == 0) full_th = (unsigned char*)malloc(width * height);

        MPI_Gatherv(th_uc,      pixCount, MPI_UNSIGNED_CHAR,
                    full_th, recvCounts, recvDispls, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);

        free(th_uc);

        // 4. Write PNG on rank 0
        if(rank == 0) {
            writePNG("thresh_debug_20.png", full_th, width, height, 1);
            free(full_th);
        }
    }
    // ----------------------------------------------------

    if (rank == 0) {
        writePNG("output.png", full, width, height, 1);
        free(full);
    }

    // Cleanup
    free(recvCounts);
    free(recvDispls);
    free(img);
    free(chunk);
    free(rgbf);
    free(outc);
    free(counts);
    free(displs);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_smooth);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_bin);
    MPI_Finalize();
    return 0;
}