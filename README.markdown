# CUDA QA Demo Programs

This repository contains simple CUDA programs to demonstrate GPU programming and automated testing for a Quality Assurance (QA) role. The programs showcase core CUDA concepts like kernel execution, memory management, and parallelism, with a Python script to automate testing.

## Prerequisites
- NVIDIA GPU with CUDA support (check [CUDA GPUs list](https://developer.nvidia.com/cuda-gpus)).
- CUDA Toolkit installed (download from [NVIDIA](https://developer.nvidia.com/cuda-downloads)).
- Python 3.x for running the test script.
- nvcc compiler (included in CUDA Toolkit).
- Linux or Windows environment.

## Programs
1. **vectorAdd.cu**: Performs element-wise addition of two vectors on the GPU.
2. **matrixMul.cu**: Implements matrix multiplication using CUDA for parallel computation.
3. **arraySumReduction.cu**: Computes the sum of an array using CUDA's reduction pattern with shared memory.

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

## Demo Instructions
- **Showcase on GitHub**: Upload this repository to your GitHub. Include this README and ensure all files are present.
- **During Interview**:
  - Explain each program's purpose (e.g., vectorAdd demonstrates basic CUDA kernel execution).
  - Run `test_cuda_programs.py` to show automated testing.
  - Discuss how the Python script checks for successful execution, reflecting QA automation skills.
  - Highlight CUDA concepts like threads, blocks, and memory management used in the code.
- **Key Points to Emphasize**:
  - Familiarity with CUDA C/C++ and GPU programming.
  - Ability to write automated tests (Python script).
  - Understanding of QA processes like verification and error handling.

## File Structure
- `vectorAdd.cu`: Vector addition program.
- `matrixMul.cu`: Matrix multiplication program.
- `arraySumReduction.cu`: Array sum reduction program.
- `test_cuda_programs.py`: Python script for automated testing.
- `README.md`: This file.

## Notes
- Ensure your GPU drivers are up to date.
- The test script assumes all `.cu` files are in the same directory.
- For Windows, replace `./` with the executable name in the test script (e.g., `vectorAdd.exe`).