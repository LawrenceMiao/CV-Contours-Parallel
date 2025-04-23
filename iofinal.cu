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
        int grayOffset = y * width + x;

        unsigned char r = d_input[rgbOffset];
        unsigned char g = d_input[rgbOffset + 1];
        unsigned char b = d_input[rgbOffset + 2];

        // Convert to grayscale using luminance formula
        d_output[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
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

                // Handle boundary conditions
                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 2][kx + 2] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = (unsigned char)sum;
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

                // Handle boundary conditions
                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = sum;
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

                // Handle boundary conditions
                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = sum;
    }
}

// CUDA kernel to compute gradient magnitude and direction
__global__ void gradientKernel(float *d_gradient_x, float *d_gradient_y, float *d_magnitude, float *d_direction, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int idx = y * width + x;
        float gx = d_gradient_x[idx];
        float gy = d_gradient_y[idx];

        // Compute magnitude
        d_magnitude[idx] = sqrtf(gx * gx + gy * gy);

        // Compute direction (in radians)
        d_direction[idx] = atan2f(gy, gx);
    }
}

// CUDA kernel for non-maximum suppression
__global__ void nonMaxSuppressionKernel(float *d_magnitude, float *d_direction, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int idx = y * width + x;
        float mag = d_magnitude[idx];
        float dir = d_direction[idx];

        // Convert direction to degrees and normalize to 0-180
        float dirDegrees = dir * 180.0f / 3.14159265358979323846f;
        if (dirDegrees < 0)
            dirDegrees += 180.0f;

        // Round to nearest 45 degrees
        int angle = ((int)(dirDegrees + 22.5f) % 180) / 45;

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

        // Ensure neighbors are within bounds
        int x1 = min(width - 1, max(0, x + nx1));
        int y1 = min(height - 1, max(0, y + ny1));
        int x2 = min(width - 1, max(0, x + nx2));
        int y2 = min(height - 1, max(0, y + ny2));

        float mag1 = d_magnitude[y1 * width + x1];
        float mag2 = d_magnitude[y2 * width + x2];

        // Apply non-maximum suppression
        if (mag >= mag1 && mag >= mag2)
        {
            // Keep this pixel only if it's a local maximum
            // Apply thresholding (can be adjusted)
            const float highThreshold = 50.0f;
            if (mag > highThreshold)
            {
                d_output[idx] = 255; // Strong edge
            }
            else
            {
                d_output[idx] = 0; // Non-edge
            }
        }
        else
        {
            d_output[idx] = 0; // Suppress non-maximum
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <input_png> <num_images>\n", argv[0]);
        return 1;
    }

    //--------------------------------------------------------------------
    // MPI start-up ------------------------------------------------------
    //--------------------------------------------------------------------
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *inputPNG = argv[1];
    int totalImages      = atoi(argv[2]);
    if (totalImages <= 0) {
        if (rank==0) fprintf(stderr,"num_images must be > 0\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    if (totalImages < size) {
        if (rank==0) fprintf(stderr,
            "Warning: num_images < #ranks – some ranks will sit idle\n");
    }

    //--------------------------------------------------------------------
    // Rank-0 loads PNG  -------------------------------------------------
    //--------------------------------------------------------------------
    RGBImage rgbImg = {0};
    if (rank == 0) {
        printf("Rank 0: reading %s\n", inputPNG);
        rgbImg = readPNG(inputPNG);
        if (rgbImg.width==0 || rgbImg.height==0) {
            fprintf(stderr,"Rank 0: failed to read %s\n", inputPNG);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    // Broadcast dimensions
    int width=rgbImg.width, height=rgbImg.height;
    MPI_Bcast(&width ,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&height,1,MPI_INT,0,MPI_COMM_WORLD);

    // Allocate / broadcast RGB bytes
    size_t rgbBytes = width*height*3;
    if (rank!=0) rgbImg.data = (unsigned char*)malloc(rgbBytes);
    MPI_Bcast(rgbImg.data, rgbBytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    //--------------------------------------------------------------------
    // Each rank decides how many “images” it must process ---------------
    //--------------------------------------------------------------------
    int baseJobs   = totalImages / size;
    int remainder  = totalImages % size;
    int myJobs     = baseJobs + (rank < remainder ? 1 : 0);

    if (myJobs==0) {
        printf("Rank %d: no work (idle)\n", rank);
        MPI_Finalize();
        return 0;
    }
    printf("Rank %d: will process %d image(s)\n", rank, myJobs);

    //--------------------------------------------------------------------
    // Set GPU for this rank --------------------------------------------
    //--------------------------------------------------------------------
    int devCount=0;  cudaGetDeviceCount(&devCount);
    if (devCount==0) { fprintf(stderr,"Rank %d: no CUDA devices\n",rank);
                       MPI_Abort(MPI_COMM_WORLD,1); }
    cudaSetDevice(rank % devCount);

    //--------------------------------------------------------------------
    // Allocate host / device buffers that we reuse for all jobs ---------
    //--------------------------------------------------------------------
    size_t grayBytes = width*height;
    size_t floatBytes= grayBytes*sizeof(float);

    GrayImage grayImg ={width,height,(unsigned char*)malloc(grayBytes)};
    GrayImage blurredImg={width,height,(unsigned char*)malloc(grayBytes)};
    GrayImage edgesImg ={width,height,(unsigned char*)malloc(grayBytes)};

    GradientImage gradImg;
    gradImg.width=width; gradImg.height=height;
    gradImg.magnitude=(float*)malloc(floatBytes);
    gradImg.direction=(float*)malloc(floatBytes);

    unsigned char *d_rgb, *d_gray, *d_blur, *d_edges;
    float *d_gx,*d_gy,*d_mag,*d_dir;
    cudaMalloc(&d_rgb,rgbBytes);
    cudaMalloc(&d_gray, grayBytes);
    cudaMalloc(&d_blur, grayBytes);
    cudaMalloc(&d_edges,grayBytes);
    cudaMalloc(&d_gx,floatBytes);
    cudaMalloc(&d_gy,floatBytes);
    cudaMalloc(&d_mag,floatBytes);
    cudaMalloc(&d_dir,floatBytes);

    dim3 blockDim(16,16);
    dim3 gridDim((width+15)/16,(height+15)/16);

    //--------------------------------------------------------------------
    // Timing start (only once) -----------------------------------------
    //--------------------------------------------------------------------
    uint64_t t0=0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) t0=clock_now();

    //--------------------------------------------------------------------
    // Main processing loop ---------------------------------------------
    //--------------------------------------------------------------------
    for(int job=0; job<myJobs; ++job)
    {
        // copy RGB once per job
        cudaMemcpy(d_rgb, rgbImg.data, rgbBytes, cudaMemcpyHostToDevice);

        // 1. RGB -> Gray
        rgbToGrayscaleKernel<<<gridDim,blockDim>>>(d_rgb,d_gray,width,height,0,height);
        cudaDeviceSynchronize();

        // 2. Gaussian blur
        gaussianBlurKernel  <<<gridDim,blockDim>>>(d_gray,d_blur,width,height,0,height);
        cudaDeviceSynchronize();

        // 3. Sobel X / Y
        sobelXKernel<<<gridDim,blockDim>>>(d_blur,d_gx,width,height,0,height);
        sobelYKernel<<<gridDim,blockDim>>>(d_blur,d_gy,width,height,0,height);
        cudaDeviceSynchronize();

        // 4. Magnitude + direction
        gradientKernel<<<gridDim,blockDim>>>(d_gx,d_gy,d_mag,d_dir,
                                             width,height,0,height);
        cudaDeviceSynchronize();

        // 5. Non-max suppression (writes uchar edges to d_edges)
        nonMaxSuppressionKernel<<<gridDim,blockDim>>>(d_mag,d_dir,d_edges,
                                                      width,height,0,height);
        cudaDeviceSynchronize();

        // copy result back (optional – here we only copy first job per rank)
        if (job==0) {
            cudaMemcpy(edgesImg.data, d_edges, grayBytes, cudaMemcpyDeviceToHost);

            // each rank writes its own PNG for first job
            char fname[128];
            sprintf(fname,"output_rank%d.png", rank);
            writeGrayPNG(fname,&edgesImg);
            printf("Rank %d: wrote %s\n", rank, fname);
        }
    }

    //--------------------------------------------------------------------
    // Timing end --------------------------------------------------------
    //--------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) {
        uint64_t t1=clock_now();
        printf("Total cycles (all ranks finished): %lu\n", t1 - t0);
    }

    /* ==============================================================
    ==  MPI-IO: append all images to one binary file (results.bin)
    ==  Layout:
    ==  int32  width,  int32 height,  int32 nImages
    ==  then  nImages × (width*height) bytes, in image-index order
    ============================================================== */

    MPI_File fh;
    MPI_Status mpistatus;
    const char *binName = "results.bin";

    MPI_File_open(MPI_COMM_WORLD, binName,
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &fh);

    /* --- header (only rank 0 writes once) ------------------------ */
    if (rank == 0) {
        int hdr[3] = { width, height, totalImages };
        MPI_File_write(fh, hdr, 3, MPI_INT, &mpistatus);
    }
    MPI_Barrier(MPI_COMM_WORLD);             /* make sure header done */

    /* --- compute my global first-image index --------------------- */
    int myFirstIdx = rank * baseJobs + (rank < remainder ? rank : remainder);
    /* bytes per image                                             */
    MPI_Offset imgBytes   = (MPI_Offset)width * height;
    MPI_Offset headerSize = 3 * sizeof(int);

    /* --- host buffer once per image ------------------------------ */
    unsigned char *hostEdge = (unsigned char*)malloc(imgBytes);

    for (int j = 0; j < myJobs; ++j) {

        /* copy current edge map (d_edges already holds job j) */
        cudaMemcpy(hostEdge, d_edges, imgBytes, cudaMemcpyDeviceToHost);

        /* file offset for this image */
        MPI_Offset offset = headerSize + (MPI_Offset)(myFirstIdx + j) * imgBytes;

        /* write synchronously (independent) */
        MPI_File_write_at(fh, offset,
                            hostEdge, imgBytes,
                            MPI_UNSIGNED_CHAR, &mpistatus);
    }
    free(hostEdge);
    MPI_File_close(&fh);
    if (rank==0) printf("All ranks: wrote %s with %d images\n", binName,totalImages);
   

    //--------------------------------------------------------------------
    // Cleanup -----------------------------------------------------------
    //--------------------------------------------------------------------
    free(grayImg.data);  free(blurredImg.data);  free(edgesImg.data);
    free(gradImg.magnitude);  free(gradImg.direction);
    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_blur); cudaFree(d_edges);
    cudaFree(d_gx);  cudaFree(d_gy);   cudaFree(d_mag);  cudaFree(d_dir);
    free(rgbImg.data);

    MPI_Finalize();
    return 0;
}
