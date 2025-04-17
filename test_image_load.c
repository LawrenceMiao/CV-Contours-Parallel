#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>

/* Function to load PNG image using libpng */
unsigned char* readPNG(const char* filename, int* width, int* height, int* channels)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    /* Initialize libpng read struct */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(png_ptr == NULL) {
        fprintf(stderr, "png_create_read_struct failed\n");
        fclose(fp);
        return NULL;
    }

    /* Initialize info struct */
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if(info_ptr == NULL) {
        fprintf(stderr, "png_create_info_struct failed\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }

    /* Set up error handling using setjmp */
    if(setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during init_io\n");
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

    /* Transformations for palette or grayscale images */
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

    /* Update info struct to reflect transformations */
    png_read_update_info(png_ptr, info_ptr);
    *width    = png_get_image_width(png_ptr, info_ptr);
    *height   = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);

    size_t rowbytes = (*width) * (*channels) * sizeof(unsigned char);
    unsigned char* data = (unsigned char*)malloc((*height) * rowbytes);
    if(data == NULL) {
        fprintf(stderr, "Error allocating memory for image data\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_bytep* row_pointers = (png_bytep*) malloc((*height) * sizeof(png_bytep));
    if(row_pointers == NULL) {
        fprintf(stderr, "Error allocating memory for row pointers\n");
        free(data);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    for(int y = 0; y < *height; y++) {
        row_pointers[y] = data + y * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);

    free(row_pointers);
    fclose(fp);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    return data;
}

/* Function to write PNG image data to a file */
int writePNG(const char* filename, unsigned char* data, int width, int height, int channels)
{
    FILE *fp = fopen(filename, "wb");
    if(fp == NULL) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(png_ptr == NULL) {
        fclose(fp);
        fprintf(stderr, "png_create_write_struct failed\n");
        return -1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if(info_ptr == NULL) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        fprintf(stderr, "png_create_info_struct failed\n");
        return -1;
    }

    if(setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        fprintf(stderr, "Error during writing PNG\n");
        return -1;
    }

    png_init_io(png_ptr, fp);

    int color_type;
    if(channels == 1)
        color_type = PNG_COLOR_TYPE_GRAY;
    else if(channels == 3)
        color_type = PNG_COLOR_TYPE_RGB;
    else if(channels == 4)
        color_type = PNG_COLOR_TYPE_RGBA;
    else {
        fprintf(stderr, "Unsupported channel count: %d\n", channels);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return -1;
    }

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, /* bit depth */
                 color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    size_t rowbytes = width * channels * sizeof(unsigned char);
    png_bytep* row_pointers = (png_bytep*) malloc(height * sizeof(png_bytep));
    if(row_pointers == NULL) {
        fprintf(stderr, "Failed to allocate row pointers for writing\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return -1;
    }

    for (int y = 0; y < height; y++) {
        row_pointers[y] = data + y * rowbytes;
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);

    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return 0;
}

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 2) {
        if(rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <size> ./test_image_load images/input.png [threshold=50]\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const char* inputFile = argv[1];
    const char* outputFile = "output.png";  /* Output file name */
    unsigned char thresholdVal = 50; /* default threshold; not used in this example */
    if(argc >= 3) {
        thresholdVal = (unsigned char) atoi(argv[2]);
    }

    int width = 0, height = 0, channels = 0;
    unsigned char* hostImage = NULL;
    if(rank == 0) {
        hostImage = readPNG(inputFile, &width, &height, &channels);
        if(hostImage == NULL) {
            fprintf(stderr, "Error reading PNG file!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        printf("Image loaded. Width=%d, Height=%d, Channels=%d\n", width, height, channels);

        /* Write the image out to a new file so you can verify it saved correctly */
        if(writePNG(outputFile, hostImage, width, height, channels) == 0) {
            printf("Successfully wrote output image to %s\n", outputFile);
        } else {
            fprintf(stderr, "Failed to write output image to %s\n", outputFile);
        }

        free(hostImage);
    }

    MPI_Finalize();
    return 0;
}
