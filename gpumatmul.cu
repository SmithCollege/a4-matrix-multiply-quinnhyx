#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define size 200

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width){
int row=blockIdx.y*blockDim.y+threadIdx.y;
int col=blockIdx.x*blockDim.x+threadIdx.x;

if(row<Width && col<Width){
	float sum = 0;
	for(int k=0;k<Width;k++){
		sum+=d_M[row*Width+k]*d_N[k*Width+col];
	}
	d_P[row*Width+col]=sum;
}
}

int main(){
  float* x =(float*) malloc(sizeof(float) * size * size);
  float* y =(float*) malloc(sizeof(float) * size * size);
  float* z =(float*) malloc(sizeof(float) * size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  }

  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x,sizeof(float)*size*size);
  cudaMalloc(&d_y,sizeof(float)*size*size);
  cudaMalloc(&d_z,sizeof(float)*size*size);

  cudaMemcpy(d_x, x,sizeof(float)*size*size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y,sizeof(float)*size*size, cudaMemcpyHostToDevice);
  
  double t0=get_clock();
  dim3 dimBlock(32, 32); // Block of 32x32 threads
  dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x, (size + dimBlock.y - 1) / dimBlock.y);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, size);
  cudaDeviceSynchronize();
  double t1=get_clock();

  cudaMemcpy(z, d_z, sizeof(float)*size*size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }

  printf("time per call: %f ns\n", (1000000000.0*(t1-t0)));
  
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(x);
  free(y);
  free(z);

  return 0;
}