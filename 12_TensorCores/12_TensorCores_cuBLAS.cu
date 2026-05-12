/*
 * cuBLAS Tensor Core TF32 vs cuBLAS FP32 matrix multiplication exercise
 *
 * This program computes batched matrix multiply-accumulates:
 *
 *     C = A * B + C
 *
 * Matrix dimensions:
 *     A: M x K, FP32
 *     B: K x N, FP32
 *     C: M x N, FP32 input/output accumulator
 *
 * This version replaces the custom Tensor Core WMMA kernel and the custom CUDA
 * ALU kernel with cuBLAS GEMM calls. It intentionally does not use cuBLAS batch
 * routines. Instead, it keeps an explicit host-side loop over the J matrices and
 * launches one cublasGemmEx call per matrix.
 *
 * Two cuBLAS execution modes are compared:
 *     1. Tensor Core path: cublasGemmEx with TF32 Tensor Core compute
 *     2. ALU / pedantic FP32 path: cublasGemmEx with full FP32 pedantic compute
 *
 * The matrices are stored in row-major order in this exercise. cuBLAS expects
 * column-major matrices, so the GEMM call computes the equivalent transposed
 * problem without physically transposing any data:
 *
 *     row-major C = A * B + C
 *
 * is represented to cuBLAS as
 *
 *     column-major C^T = B^T * A^T + C^T
 *
 * Compile example:
 *     nvcc -O3 -arch=sm_80 12_TensorCores_CUBLAS_TF32_vs_FP32.cu -lcublas -o cublas_tf32_vs_fp32
 *
 * Use the SM version of your GPU if it is newer, for example sm_86, sm_89,
 * or sm_90. TF32 Tensor Cores require NVIDIA Ampere architecture, sm_80 or newer.
 *
 * Run:
 *     ./cublas_tf32_vs_fp32
 */

 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #include <cublas_v2.h>
 
 #include <cmath>
 #include <cstdlib>
 #include <iomanip>
 #include <iostream>
 #include <random>
 #include <vector>
 
 using namespace std;
 
 #define M 9182
 #define N 4096
 #define K 4096
 #define J 16
 
 #define SIZE_A (M * K)
 #define SIZE_B (K * N)
 #define SIZE_C (M * N)
 
 /*
  * Full CPU reference for a 2048x2048x2048 GEMM is intentionally disabled by
  * default because it is extremely slow. Set this to 1 if you want the original
  * style CPU correctness check for matrix 0.
  */
 #define RUN_CPU_REFERENCE 0
 
 #define TENSOR_CORE_COMPUTE_TYPE CUBLAS_COMPUTE_32F_FAST_TF32 //CUBLAS_COMPUTE_32F_FAST_16F 
 #define ALU_COMPUTE_TYPE CUBLAS_COMPUTE_32F_PEDANTIC
 
 /*
  * CUDA error checking wrapper.
  */
 #define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
     if (code != cudaSuccess) {
         cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
         if (abort) {
             exit(code);
         }
     }
 }
 
 const char* cublasStatusToString(cublasStatus_t status) {
     switch (status) {
         case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
         case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
         case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
         case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
         case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
         case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
         case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
         case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
 #if defined(CUBLAS_STATUS_NOT_SUPPORTED)
         case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
 #endif
 #if defined(CUBLAS_STATUS_LICENSE_ERROR)
         case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
 #endif
         default: return "Unknown cuBLAS status";
     }
 }
 
 #define cublasErrCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
 inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true) {
     if (code != CUBLAS_STATUS_SUCCESS) {
         cout << "cuBLASassert: " << cublasStatusToString(code) << " " << file << " " << line << endl;
         if (abort) {
             exit(code);
         }
     }
 }
 
 /*
  * Simple CPU reference implementation for one matrix.
  * This is intentionally straightforward and not optimized.
  */
 void matmulCPU(const float* A, const float* B, const float* C, float* output) {
     for (int row = 0; row < M; ++row) {
         for (int col = 0; col < N; ++col) {
             float sum = C[static_cast<size_t>(row) * N + col];
 
             for (int kk = 0; kk < K; ++kk) {
                 float a = A[static_cast<size_t>(row) * K + kk];
                 float b = B[static_cast<size_t>(kk) * N + col];
                 sum += a * b;
             }
 
             output[static_cast<size_t>(row) * N + col] = sum;
         }
     }
 }
 
 /*
  * Frobenius norm of the difference between two MxN matrices:
  *     ||X - Y||_F = sqrt(sum_ij (X_ij - Y_ij)^2)
  */
 double frobeniusNormDiff(const float* reference, const float* result, size_t size) {
     double sumSquares = 0.0;
 
     for (size_t i = 0; i < size; ++i) {
         double diff = static_cast<double>(reference[i]) - static_cast<double>(result[i]);
         sumSquares += diff * diff;
     }
 
     return sqrt(sumSquares);
 }
 
 double frobeniusNorm(const float* matrix, size_t size) {
     double sumSquares = 0.0;
 
     for (size_t i = 0; i < size; ++i) {
         double value = static_cast<double>(matrix[i]);
         sumSquares += value * value;
     }
 
     return sqrt(sumSquares);
 }
 
 void printComparison(const char* name, const float* reference, const float* result, size_t size) {
     double diffNorm = frobeniusNormDiff(reference, result, size);
     double refNorm = frobeniusNorm(reference, size);
     double relative = diffNorm / (refNorm + 1e-20);
 
     cout << name << endl;
     cout << "  Frobenius norm ||result - reference||_F: " << diffNorm << endl;
     cout << "  Relative Frobenius error:               " << relative << endl;
 }
 
 void fillRandomBatch(vector<float>& hostA, vector<float>& hostB, vector<float>& hostC) {
     mt19937 rng(1234);
     uniform_real_distribution<float> dist(-100.0f, 100.0f);
 
     for (int batch = 0; batch < J; ++batch) {
         size_t baseA = static_cast<size_t>(batch) * SIZE_A;
         size_t baseB = static_cast<size_t>(batch) * SIZE_B;
         size_t baseC = static_cast<size_t>(batch) * SIZE_C;
 
         for (int i = 0; i < SIZE_A; ++i) {
             hostA[baseA + i] = dist(rng);
         }
         for (int i = 0; i < SIZE_B; ++i) {
             hostB[baseB + i] = dist(rng);
         }
         for (int i = 0; i < SIZE_C; ++i) {
             hostC[baseC + i] = dist(rng);
         }
 
         /*
          * Make the per-matrix uniqueness explicit and deterministic.
          */
         float batchValue = (J > 1)
                                ? -1.0f + 2.0f * static_cast<float>(batch) /
                                              static_cast<float>(J - 1)
                                : 0.0f;
         hostA[baseA + 0] = batchValue;
         hostA[baseA + 1] = -batchValue;
     }
 }
 
 
 float timeCublasTensorCoreBatch(cublasHandle_t handle,
                       const float* deviceA,
                       const float* deviceB,
                       float* deviceC,
                       int batchCount) {
     cudaEvent_t start;
     cudaEvent_t stop;
     gpuErrCheck(cudaEventCreate(&start));
     gpuErrCheck(cudaEventCreate(&stop));
 
     gpuErrCheck(cudaEventRecord(start));

     //TODO
 
     gpuErrCheck(cudaEventRecord(stop));
     gpuErrCheck(cudaEventSynchronize(stop));
 
     float elapsedMs = 0.0f;
     gpuErrCheck(cudaEventElapsedTime(&elapsedMs, start, stop));
 
     gpuErrCheck(cudaEventDestroy(start));
     gpuErrCheck(cudaEventDestroy(stop));
 
     return elapsedMs;
 }

float timeCublasAluBatch(cublasHandle_t handle,
                        const float* deviceA,
                        const float* deviceB,
                        float* deviceC,
                        int batchCount) {

    cudaEvent_t start;
    cudaEvent_t stop;
    gpuErrCheck(cudaEventCreate(&start));
    gpuErrCheck(cudaEventCreate(&stop));

    gpuErrCheck(cudaEventRecord(start));

    //TODO

    gpuErrCheck(cudaEventRecord(stop));
    gpuErrCheck(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    gpuErrCheck(cudaEventElapsedTime(&elapsedMs, start, stop));

    gpuErrCheck(cudaEventDestroy(start));
    gpuErrCheck(cudaEventDestroy(stop));

    return elapsedMs;
}
 
 int main(void) {
     int device = 0;
     cudaDeviceProp prop;
     gpuErrCheck(cudaGetDevice(&device));
     gpuErrCheck(cudaGetDeviceProperties(&prop, device));
 
     cout << "GPU: " << prop.name << endl;
     cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
 
     if (prop.major < 8) {
         cout << "This exercise needs TF32 Tensor Core support, i.e. compute capability 8.0 or newer." << endl;
         return 0;
     }
 
     cout << endl << "Allocating host matrices. This version intentionally keeps CPU data generation." << endl;
     cout << "Allocating " << J << " matrices" << endl;
     cout << "Matrix A size: " << "M=" << M << " K=" << K << endl;
     cout << "Matrix B size: " << "K=" << K << " N=" << N << endl;
     cout << "Matrix C size: " << "M=" << M << " N=" << N << endl;
 
     // Allocate and initialize J distinct random matrices on the host.
     vector<float> hostA(static_cast<size_t>(J) * SIZE_A);
     vector<float> hostB(static_cast<size_t>(J) * SIZE_B);
     vector<float> hostC(static_cast<size_t>(J) * SIZE_C);
     fillRandomBatch(hostA, hostB, hostC);
 
     vector<float> hostOutputTensorCore(SIZE_C);
     vector<float> hostOutputALU(SIZE_C);
 #if RUN_CPU_REFERENCE
     vector<float> hostOutputCPU(SIZE_C);
 #endif
 
     const int checkBatch = 0;
     const float* checkA = hostA.data() + static_cast<size_t>(checkBatch) * SIZE_A;
     const float* checkB = hostB.data() + static_cast<size_t>(checkBatch) * SIZE_B;
     const float* checkC = hostC.data() + static_cast<size_t>(checkBatch) * SIZE_C;
 
 #if RUN_CPU_REFERENCE
     cout << "Computing CPU reference for matrix " << checkBatch << ". This is intentionally slow." << endl;
     matmulCPU(checkA, checkB, checkC, hostOutputCPU.data());
 #endif
 
     // Allocate device memory for the full batch.
     float* deviceA;
     float* deviceB;
     float* deviceCTensorCore;
     float* deviceCALU;
 
     size_t bytesA = static_cast<size_t>(J) * SIZE_A * sizeof(float);
     size_t bytesB = static_cast<size_t>(J) * SIZE_B * sizeof(float);
     size_t bytesC = static_cast<size_t>(J) * SIZE_C * sizeof(float);
 
     gpuErrCheck(cudaMalloc((void**)&deviceA, bytesA));
     gpuErrCheck(cudaMalloc((void**)&deviceB, bytesB));
     gpuErrCheck(cudaMalloc((void**)&deviceCTensorCore, bytesC));
     gpuErrCheck(cudaMalloc((void**)&deviceCALU, bytesC));
 
     // Copy the benchmark data to the GPU once. This is intentionally not timed.
     gpuErrCheck(cudaMemcpy(deviceA, hostA.data(), bytesA, cudaMemcpyHostToDevice));
     gpuErrCheck(cudaMemcpy(deviceB, hostB.data(), bytesB, cudaMemcpyHostToDevice));
     gpuErrCheck(cudaMemcpy(deviceCTensorCore, hostC.data(), bytesC, cudaMemcpyHostToDevice));
     gpuErrCheck(cudaMemcpy(deviceCALU, hostC.data(), bytesC, cudaMemcpyHostToDevice));
 
     cublasHandle_t handle;
     cublasErrCheck(cublasCreate(&handle));
 
     // Warm up both paths once so the benchmark does not include one-time setup effects.
     for (int batch = 0; batch < J; ++batch) {
         const float* matrixA = deviceA + static_cast<size_t>(batch) * SIZE_A;
         const float* matrixB = deviceB + static_cast<size_t>(batch) * SIZE_B;
         float* matrixCTensorCore = deviceCTensorCore + static_cast<size_t>(batch) * SIZE_C;
         float* matrixCALU = deviceCALU + static_cast<size_t>(batch) * SIZE_C;

         // TODO
 

     }
     gpuErrCheck(cudaDeviceSynchronize());
 
     // The GEMM calls update C in place. Reset C after warmup so the timed run
     // starts from the same initial accumulator values.
     gpuErrCheck(cudaMemcpy(deviceCTensorCore, hostC.data(), bytesC, cudaMemcpyHostToDevice));
     gpuErrCheck(cudaMemcpy(deviceCALU, hostC.data(), bytesC, cudaMemcpyHostToDevice));
 
     // Timed benchmark over J different matrices.
     float tensorCoreMs = timeCublasTensorCoreBatch(handle, deviceA, deviceB, deviceCTensorCore, J);
     float aluMs = timeCublasAluBatch(handle, deviceA, deviceB, deviceCALU, J);
 
     // Copy the result for one matrix back to the CPU for correctness reporting.
     gpuErrCheck(cudaMemcpy(hostOutputTensorCore.data(),
                            deviceCTensorCore + static_cast<size_t>(checkBatch) * SIZE_C,
                            SIZE_C * sizeof(float), cudaMemcpyDeviceToHost));
     gpuErrCheck(cudaMemcpy(hostOutputALU.data(),
                            deviceCALU + static_cast<size_t>(checkBatch) * SIZE_C,
                            SIZE_C * sizeof(float), cudaMemcpyDeviceToHost));
 
     cout << fixed << setprecision(10) << endl;
 #if RUN_CPU_REFERENCE
     printComparison("cuBLAS Tensor Core TF32 result compared with CPU reference:",
                     hostOutputCPU.data(), hostOutputTensorCore.data(), SIZE_C);
     printComparison("cuBLAS full FP32 pedantic result compared with CPU reference:",
                     hostOutputCPU.data(), hostOutputALU.data(), SIZE_C);
 #else
     printComparison("cuBLAS Tensor Core TF32 result compared with cuBLAS full FP32 pedantic result:",
                     hostOutputALU.data(), hostOutputTensorCore.data(), SIZE_C);
     cout << "  CPU reference skipped because RUN_CPU_REFERENCE is 0." << endl;
 #endif
 
     cout << endl << "M=" << M << " N=" << N << " K=" << K << " J=" << J << endl;
     cout << "cuBLAS calls per path: " << J << endl;
 
     // Show a few values so students can see the outputs element-wise.
     cout << endl << "First 8 C/output elements for checked matrix " << checkBatch << ":" << endl;
 #if RUN_CPU_REFERENCE
     cout << "  CPU:                ";
     for (int i = 0; i < 8; ++i) cout << hostOutputCPU[i] << " ";
     cout << endl;
 #else
     cout << "  Initial C:          ";
     for (int i = 0; i < 8; ++i) cout << checkC[i] << " ";
     cout << endl;
 #endif
 
     cout << "  cuBLAS TF32:        ";
     for (int i = 0; i < 8; ++i) cout << hostOutputTensorCore[i] << " ";
     cout << endl;
 
     cout << "  cuBLAS FP32 pedant: ";
     for (int i = 0; i < 8; ++i) cout << hostOutputALU[i] << " ";
     cout << endl;
 
     double tensorCoreAvgMs = static_cast<double>(tensorCoreMs) / static_cast<double>(J);
     double aluAvgMs = static_cast<double>(aluMs) / static_cast<double>(J);
     double speedup = aluAvgMs / (tensorCoreAvgMs + 1e-20);
 
     cout << endl << "Execution-time benchmark over " << J << " different random matrices:" << endl;
     cout << "  Timing scope: one host-side loop per path, with one cublasGemmEx call per matrix;" << endl;
     cout << "                includes device-side GEMM execution and global-memory reads/writes." << endl;
     cout << "  Not timed:    CPU random-number generation, one-time host-to-device input upload," << endl;
     cout << "                and C reset after warmup." << endl;
     cout << endl;
     cout << "  cuBLAS Tensor Core TF32 total time:     " << tensorCoreMs << " ms" << endl;
     cout << "  cuBLAS full FP32 pedantic total time:   " << aluMs << " ms" << endl;
     cout << "  cuBLAS Tensor Core TF32 average/matrix: " << tensorCoreAvgMs << " ms" << endl;
     cout << "  cuBLAS full FP32 average/matrix:        " << aluAvgMs << " ms" << endl;
     cout << "  FP32 / TF32 time:                       " << speedup << "x" << endl;
 
     if (speedup > 1.0) {
         cout << "  TF32 Tensor Core is faster by:          " << speedup << "x" << endl;
     } else {
         cout << "  Full FP32 pedantic is faster by:        " << (1.0 / speedup) << "x" << endl;
     }
 
     // Free device and cuBLAS resources.
     cublasErrCheck(cublasDestroy(handle));
     gpuErrCheck(cudaFree(deviceA));
     gpuErrCheck(cudaFree(deviceB));
     gpuErrCheck(cudaFree(deviceCTensorCore));
     gpuErrCheck(cudaFree(deviceCALU));
 
     return 0;
 }
 