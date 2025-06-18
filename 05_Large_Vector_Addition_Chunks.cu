#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TOTAL_SIZE (1024 * 1024 * 1024)  // Total elements
#define CHUNK_SIZE (1024 * 1024 * 128)   // Elements per chunk
#define BLOCK_SIZE 1024                 // Threads per block

// CUDA Kernel
__global__ void vectorAdd(int* a, int* b, int* c, int chunk_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < chunk_size) {
        c[index] = a[index] + b[index];
    }
}

// Host function to fill array with random ints
void random_ints(int* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = rand() % 100;
    }
}

int main() {
    int* d_a, * d_b, * d_c;
    int* chunk_a, * chunk_b, * chunk_c;
    size_t chunkSizeBytes = CHUNK_SIZE * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_a, chunkSizeBytes);
    cudaMalloc((void**)&d_b, chunkSizeBytes);
    cudaMalloc((void**)&d_c, chunkSizeBytes);
    printf("\nHello 01\n");

    // Allocate host memory
    chunk_a = (int*)malloc(chunkSizeBytes);
    chunk_b = (int*)malloc(chunkSizeBytes);
    chunk_c = (int*)malloc(chunkSizeBytes);

    for (long long offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE) {
        int currentChunkSize = (TOTAL_SIZE - offset) < CHUNK_SIZE ? (TOTAL_SIZE - offset) : CHUNK_SIZE;
        int numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("\nOffset %lld\n", offset);

        random_ints(chunk_a, currentChunkSize);
        random_ints(chunk_b, currentChunkSize);

        cudaMemcpy(d_a, chunk_a, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, chunk_b, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);

        // Launch the kernel
        vectorAdd << <numBlocks, BLOCK_SIZE >> > (d_a, d_b, d_c, currentChunkSize);

        // Copy result back
        cudaMemcpy(chunk_c, d_c, currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost);

        // Optional: Process chunk_c here
    }

    printf("\nFirst elements of the result vector:\n");
    for (int i = 0; i < 100; i++) {
        printf("Element %d    %d + %d = %d\n", i, chunk_a[i], chunk_b[i], chunk_c[i]);
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(chunk_a);
    free(chunk_b);
    free(chunk_c);

    return 0;
}
