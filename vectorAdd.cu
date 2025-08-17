#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Step 1: Allocate memory for arrays on the CPU and GPU
    const int N = 1000; // Vector size
    size_t size = N * sizeof(float);

    // Allocate memory for arrays on the CPU (host)
    float *h_A = (float *)malloc(size); // Input vector A
    float *h_B = (float *)malloc(size); // Input vector B
    float *h_C = (float *)malloc(size); // Output vector C

    // Step 2: Initialize two input arrays with random numbers
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX; // Random values between 0 and 1
        h_B[i] = rand() / (float)RAND_MAX; // Random values between 0 and 1
    }

    // Allocate memory for arrays on the GPU (device)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); // GPU memory for input vector A
    cudaMalloc(&d_B, size); // GPU memory for input vector B
    cudaMalloc(&d_C, size); // GPU memory for output vector C

    // Step 3: Copy arrays to the GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Copy A from CPU to GPU
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); // Copy B from CPU to GPU

    // Step 4: Launch the vectorAdd kernel to perform addition
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Launch kernel

    // Step 5: Copy the result back to the CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // Copy result C from GPU to CPU

    // Step 6: Verify the result by checking if C[i] = A[i] + B[i] for all elements
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            break;
        }
    }
    printf("Vector addition completed successfully!\n");

    // Step 7: Free memory
    cudaFree(d_A); // Free GPU memory for A
    cudaFree(d_B); // Free GPU memory for B
    cudaFree(d_C); // Free GPU memory for C
    free(h_A);     // Free CPU memory for A
    free(h_B);     // Free CPU memory for B
    free(h_C);     // Free CPU memory for C
