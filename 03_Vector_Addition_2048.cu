#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 2048  // Define the size of the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int* A, int* B, int* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        C[i] = A[i] + B[i];
}

int main() {
    int* A, * B, * C;            // Host vectors
    int* d_A, * d_B, * d_C;      // Device vectors
    int size = SIZE * sizeof(int);

    // Allocate and initialize host vectors
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);



    // Allocate device vectors
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }

    // Copy host vectors to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vectorAdd <<<2, 1024 >>> (d_A, d_B, d_C, SIZE);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate and print the execution time
    printf("\nExecution finished\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%d + %d = %d  \n", A[i], B[i], C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}