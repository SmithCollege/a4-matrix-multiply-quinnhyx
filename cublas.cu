#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <sys/time.h>

#define N 200

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void init(float* mat, int size){
     for(int i=0;i<size;i++){
     	   mat[i]=1;
	}
}

void cublas(){
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
     const int size = N*N;

     float* h_a = (float*)malloc(sizeof(float) * size);
     float* h_b = (float*)malloc(sizeof(float) * size);
     float* h_c = (float*)malloc(sizeof(float) * size);

     init(h_a,size);
     init(h_b,size);
     
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * size);
    cudaMalloc((void**)&d_b, sizeof(float) * size);
    cudaMalloc((void**)&d_c, sizeof(float) * size);

     cudaMemcpy(d_a, h_a, sizeof(float) * size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b, sizeof(float) * size, cudaMemcpyHostToDevice);

     cublasHandle_t handle;
    cublasCreate(&handle);
    
     double t0=get_clock();
    // Perform warmup operation with cublas
       cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, d_a, N, d_b, N,&beta, d_c, N);
	double t1 = get_clock();
	
	cudaMemcpy(h_c, d_c, sizeof(float)*size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	

     printf("Time per call: %f ns\n", (1000000000.0 * (t1 - t0)));

     cublasDestroy(handle);
     free(h_a);
     free(h_b);
     free(h_c);
     cudaFree(d_a);
     cudaFree(d_b);
     cudaFree(d_c);
}

int main(){
    cublas();
    return 0;
 }
