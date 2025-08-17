# CUDA Demo Programs

This repository contains simple CUDA program to demonstrate GPU programming, it showcases core CUDA concepts like kernel execution, memory management, and parallelism, with a Python script to automate testing.

## Prerequisites
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- Python 3.x for running the test script.
- nvcc compiler (included in CUDA Toolkit).
- Linux or Windows environment.

## Programs
1. **vectorAdd.cu**: Performs element-wise addition of two vectors on the GPU.

## How to Build and Run
1. **Compile a program**:
   ```bash
   nvcc vectorAdd.cu -o vectorAdd
   ```
   Replace `vectorAdd.cu` with the desired program name.
2. **Run a program**:
   ```bash
   ./vectorAdd
   ```
3. **Run automated tests**:
   ```bash
   python test_cuda_programs.py
   ```
   The Python script compiles and tests all programs, verifying their output.

## Understanding `vectorAdd.cu`
Here’s a breakdown of what the program does:

1. **Purpose**: Adds two arrays (`A` and `B`) element-wise to produce a result array (`C`) using the GPU.
2. **Key CUDA Concepts**:
   - **Kernel**: A function (`vectorAdd`) that runs on the GPU. It’s marked with `__global__` and executed by many threads in parallel.
   - **Threads and Blocks**: The program splits the work into small pieces (threads) organized into blocks. Each thread adds one pair of elements (e.g., `C[i] = A[i] + B[i]`).
   - **Memory Management**: Data is moved between the CPU (host) and GPU (device) using `cudaMalloc` and `cudaMemcpy`.
3. **Program Flow**:
   - Allocate memory for arrays on the CPU and GPU.
   - Initialize two input arrays with random numbers.
   - Copy arrays to the GPU.
   - Launch the `vectorAdd` kernel to perform addition.
   - Copy the result back to the CPU.
   - Verify the result by checking if `C[i] = A[i] + B[i]` for all elements.
   - Free memory.
