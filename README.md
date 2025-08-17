# CUDA QA Demo Programs

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

## File Structure
- `vectorAdd.cu`: Vector addition program.
- `test_cuda_programs.py`: Python script for automated testing.
- `README.md`: This file.
