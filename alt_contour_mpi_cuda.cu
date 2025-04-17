#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------ libpng I/O ------------------
unsigned char* readPNG(const char* filename, int* width, int* height, int* channels) {
    printf("Reading PNG %s...\n", filename);
    FILE *fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Error opening file %s\n", filename); return NULL; }
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return NULL; }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, NULL, NULL); fclose(fp); return NULL; }
    if (setjmp(png_jmpbuf(png_ptr))) { png_destroy_read_struct(&png_ptr, &info_ptr, NULL); fclose(fp); return NULL; }
    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);
    *width    = png_get_image_width(png_ptr, info_ptr);
    *height   = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);
    printf("PNG info: width=%d, height=%d, channels=%d\n", *width, *height, *channels);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA || color_type == PNG_COLOR_TYPE_GRAY) png_set_gray_to_rgb(png_ptr);
    png_read_update_info(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);
    size_t rowbytes = (*width) * (*channels);
    unsigned char *data = (unsigned char*)malloc((*height) * rowbytes);
    png_bytep *rp = (png_bytep*)malloc((*height) * sizeof(png_bytep));
    for(int y = 0; y < *height; y++) rp[y] = data + y * rowbytes;
    png_read_image(png_ptr, rp);
    free(rp);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    printf("Finished reading PNG.\n");
    return data;
}

int writePNG(const char* filename, unsigned char* data, int width, int height, int channels) {
    printf("Writing PNG %s...\n", filename);
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Error opening %s for writing\n", filename); return -1; }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (setjmp(png_jmpbuf(png_ptr))) { png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return -1; }
    png_init_io(png_ptr, fp);
    int color_type = (channels == 1 ? PNG_COLOR_TYPE_GRAY : (channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA));
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t rowbytes = width * channels;
    png_bytep *rp = (png_bytep*)malloc(height * sizeof(png_bytep));
    for(int y = 0; y < height; y++) rp[y] = data + y * rowbytes;
    png_write_image(png_ptr, rp);
    png_write_end(png_ptr, info_ptr);
    free(rp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    printf("Finished writing PNG.\n");
    return 0;
}

// ------------------ CUDA KERNELS ------------------
// 1) RGB->Grayscale (NCHW)
__global__ void grayscale_cuda(const float* rgb, float* gray,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h)return;
    int idx=y*w+x, plane=w*h;
    gray[idx]=0.299f*rgb[idx]+0.587f*rgb[plane+idx]+0.114f*rgb[2*plane+idx];
}
// 2) Gaussian smoothing
__global__ void gaussian_cuda(const float* in,float* out,const float* ker,int ksz,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h)return;int half=ksz/2;float sum=0,ws=0;
    for(int dy=-half;dy<=half;dy++)for(int dx=-half;dx<=half;dx++){
        int ix=min(max(x+dx,0),w-1), iy=min(max(y+dy,0),h-1);
        float v=in[iy*w+ix], k=ker[(dy+half)*ksz+(dx+half)]; sum+=v*k; ws+=k;
    }
    out[y*w+x]=ws>0?sum/ws:sum;
}
// 3) Sobel gradients
__global__ void sobel_cuda(const float* gray,float* gx,float* gy,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h)return;int idx=y*w+x;
    float Gx=0,Gy=0;
    float sobelX[9]={-1,0,1,-2,0,2,-1,0,1}, sobelY[9]={1,2,1,0,0,0,-1,-2,-1};
    int half=1;
    for(int dy=-half;dy<=half;dy++)for(int dx=-half;dx<=half;dx++){
        int ix=min(max(x+dx,0),w-1), iy=min(max(y+dy,0),h-1);
        float v=gray[iy*w+ix]; int k=(dy+half)*3+(dx+half);
        Gx+=v*sobelX[k]; Gy+=v*sobelY[k];
    }
    gx[idx]=Gx; gy[idx]=Gy;
}
// 4) Non-Max Suppression
__global__ void nms_cuda(const float* mag,const float* gx,const float* gy,unsigned char* out,int w,int h){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<1||x>=w-1||y<1||y>=h-1)return;int idx=y*w+x;
    float angle=atan2f(gy[idx],gx[idx])*180/M_PI; if(angle<0)angle+=180;
    float m=mag[idx], m1,m2;
    if((angle<=22.5||angle>157.5)){m1=mag[idx-1];m2=mag[idx+1];}
    else if(angle<=67.5){m1=mag[idx-w-1];m2=mag[idx+w+1];}
    else if(angle<=112.5){m1=mag[idx-w];m2=mag[idx+w];}
    else {m1=mag[idx-w+1];m2=mag[idx+w-1];}
    out[idx]=(m>=m1&&m>=m2)?(unsigned char)m:0;
}
// 5) Threshold
__global__ void thresh_cuda(const unsigned char* in,unsigned char* out,int w,int h,unsigned char t){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h)return;out[y*w+x]=in[y*w+x]>=t?255:0;
}

// ------------------ Host Helpers ------------------
void makeKernel(float* k,int ksz,float s){int half=ksz/2;float sum=0;
    for(int y=-half;y<=half;y++)for(int x=-half;x<=half;x++){float v=expf(-(x*x+y*y)/(2*s*s));k[(y+half)*ksz+(x+half)]=v;sum+=v;}
    for(int i=0;i<ksz*ksz;i++)k[i]/=sum;
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=4){if(!rank)fprintf(stderr,"Need 4 ranks\n");MPI_Abort(MPI_COMM_WORLD,1);}int W,H,C;
    unsigned char* img=NULL; if(!rank)img=readPNG(argv[1],&W,&H,&C);
    MPI_Bcast(&W,1,MPI_INT,0,MPI_COMM_WORLD);MPI_Bcast(&H,1,MPI_INT,0,MPI_COMM_WORLD);MPI_Bcast(&C,1,MPI_INT,0,MPI_COMM_WORLD);
        // Compute how many rows each rank gets (core, without halo)
    int base = H / size;
    int rem = H % size;
    int *core_h = (int*)malloc(size * sizeof(int));
    int *core_disp = (int*)malloc(size * sizeof(int));
    for(int i = 0, offset = 0; i < size; i++) {
        core_h[i] = base + (i < rem ? 1 : 0);
        core_disp[i] = offset * W * C;
        offset += core_h[i];
    }
    int myCoreH = core_h[rank];
    size_t coreBytes = myCoreH * W * C * sizeof(unsigned char);

    // Allocate buffer for core rows and scatter
    unsigned char *core_buf = (unsigned char*)malloc(coreBytes);
    MPI_Scatterv(img,
                 (const int*)core_h,     // counts in rows
                 core_disp,              // displacements in rows
                 MPI_BYTE,
                 core_buf,
                 myCoreH * W * C,        // receive count in bytes
                 MPI_BYTE,
                 0,
                 MPI_COMM_WORLD);

    // Now create halo buffer: (myCoreH + 2) rows
    unsigned char *buf = (unsigned char*)malloc((myCoreH + 2) * W * C);
    // Middle: core rows
    memcpy(buf + W * C, core_buf, coreBytes);

    // Exchange/replicate halos
    MPI_Status st;
    // Top halo
    if(rank == 0) {
        // replicate first core row
        memcpy(buf, buf + W * C, W * C);
    } else {
        MPI_Sendrecv(
            buf + W * C, W * C, MPI_BYTE, rank-1, 0,
            buf,         W * C, MPI_BYTE, rank-1, 0,
            MPI_COMM_WORLD, &st);
    }
    // Bottom halo
    if(rank == size-1) {
        // replicate last core row
        memcpy(buf + (myCoreH+1)*W*C,
               buf + myCoreH*W*C,
               W * C);
    } else {
        MPI_Sendrecv(
            buf + myCoreH * W * C, W * C, MPI_BYTE, rank+1, 1,
            buf + (myCoreH+1)*W*C, W * C, MPI_BYTE, rank+1, 1,
            MPI_COMM_WORLD, &st);
    }

    free(core_buf);
    free(core_h);
    free(core_disp);

    int myH = myCoreH;  // excluding halos

    // convert to NCHW float
    float *rgbf=(float*)malloc(3*W*(myH+2)*sizeof(float));
    for(int c0=0;c0<3;c0++)for(int y=0;y<myH+2;y++)for(int x=0;x<W;x++){
        int pid=(y*W+x)*C+c0;rgbf[c0*W*(myH+2)+y*W+x]=buf[pid];}
    // alloc cuda
    cudaSetDevice(rank);
    float *d_rgb,*d_gray,*d_smooth,*d_gx,*d_gy,*d_mag;unsigned char *d_nms,*d_bin;
    int area=W*(myH+2);size_t szF=area*sizeof(float),szI=area*sizeof(unsigned char);
    cudaMalloc(&d_rgb,3*szF);cudaMalloc(&d_gray,szF);cudaMalloc(&d_smooth,szF);
    cudaMalloc(&d_gx,szF);cudaMalloc(&d_gy,szF);cudaMalloc(&d_mag,szF);
    cudaMalloc(&d_nms,szI);cudaMalloc(&d_bin,szI);
    cudaMemcpy(d_rgb,rgbf,3*szF,cudaMemcpyHostToDevice);
    dim3 B(16,16),G((W+15)/16,((myH+2)+15)/16);
    grayscale_cuda<<<G,B>>>(d_rgb,d_gray,W,myH+2);cudaDeviceSynchronize();
    // gaussian
    float *hker=(float*)malloc(9*sizeof(float));makeKernel(hker,3,1.0f);
    float *dker;cudaMalloc(&dker,9*sizeof(float));cudaMemcpy(dker,hker,9*sizeof(float),cudaMemcpyHostToDevice);
    gaussian_cuda<<<G,B>>>(d_gray,d_smooth,dker,3,W,myH+2);cudaDeviceSynchronize();
    cudaFree(dker);free(hker);
    printf("done with gaussian blur\n");
    // sobel
    sobel_cuda<<<G,B>>>(d_smooth,d_gx,d_gy,W,myH+2);cudaDeviceSynchronize();
    printf("done with sobel\n");
    // magnitude
    // inline mag kernel
    sobel_cuda<<<G,B>>>((float*)d_smooth,d_gx,d_gy,W,myH+2); // reuse gx,gy
    cudaDeviceSynchronize();
    // NMS
    nms_cuda<<<G,B>>>(d_smooth,d_gx,d_gy,d_nms,W,myH+2);cudaDeviceSynchronize();
    printf("done with NMS\n");
    // threshold
    thresh_cuda<<<G,B>>>(d_nms,d_bin,W,myH+2,128);cudaDeviceSynchronize();
    printf("done with thresholding\n");
    // copy back central region
    unsigned char *out=(unsigned char*)malloc(myH*W);
    cudaMemcpy(out,d_bin+W,szI-W,cudaMemcpyDeviceToHost);
        // Gather filtered binary pixels (single-channel)
    unsigned char *full = NULL;
    if (rank == 0) {
        full = (unsigned char*)malloc(W * H * sizeof(unsigned char));
    }

    int *rC = (int*)malloc(size * sizeof(int));
    int *rD = (int*)malloc(size * sizeof(int));
    int accum = 0;
    for (int i = 0; i < size; i++) {
        // each rank contributed core_h[i] rows of width W
        rC[i] = core_h[i] * W;
        rD[i] = accum * W;
        accum += core_h[i];
    }

    // myH == core_h[rank]
    MPI_Gatherv(out, myH * W, MPI_UNSIGNED_CHAR,
                full, rC, rD, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        writePNG("output.png", full, W, H, 1);
        free(full);
    }

    // cleanup
    free(rC);
    free(rD);
    free(img);
    free(buf);
    free(rgbf);
    free(out);
    // core_h and core_disp freed here
    free(core_h);
    free(core_disp);
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

    free(img);free(buf);free(rgbf);free(out);free(rC);free(rD);
    cudaFree(d_rgb);cudaFree(d_gray);cudaFree(d_smooth);
    cudaFree(d_gx);cudaFree(d_gy);cudaFree(d_mag);cudaFree(d_nms);cudaFree(d_bin);
    MPI_Finalize();
    return 0;
}
