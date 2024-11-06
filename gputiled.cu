#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define size 200
#define TILE_WIDTH 32

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void tiled(float* M, float* N, float* P, int Width) {
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int m = 0; m < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Collaborative loading of M and N tiles into shared memory with bounds checking
        if (Row < Width && m * TILE_WIDTH + tx < Width) {
            subTileM[ty][tx] = M[Row * Width + m * TILE_WIDTH + tx];
        } else {
            subTileM[ty][tx] = 0.0; // Initialize to zero if out of bounds
        }

        if (Col < Width && m * TILE_WIDTH + ty < Width) {
            subTileN[ty][tx] = N[(m * TILE_WIDTH + ty) * Width + Col];
        } else {
            subTileN[ty][tx] = 0.0; // Initialize to zero if out of bounds
        }

        __syncthreads();

        // Perform matrix multiplication on the tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        }
        __syncthreads();
    }

    // Write the result to the output matrix if within bounds
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}


int main() {
    float* x = (float*)malloc(sizeof(float) * size * size);
    float* y = (float*)malloc(sizeof(float) * size * size);
    float* z = (float*)malloc(sizeof(float) * size * size);

    // Initialize matrices x and y with values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x[i * size + j] = 1; 
            y[i * size + j] = 1;
        }
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, sizeof(float) * size * size);
    cudaMalloc(&d_y, sizeof(float) * size * size);
    cudaMalloc(&d_z, sizeof(float) * size * size);

    cudaMemcpy(d_x, x, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions for tiled matrix multiplication
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);

    double t0 = get_clock();
    tiled<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, size);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    double t1 = get_clock();

    cudaMemcpy(z, d_z, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (z[i * size + j] != size) {
                printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
            }
        }
    }

    printf("Time per call: %f ns\n", (1000000000.0 * (t1 - t0)));

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);

    return 0;
}