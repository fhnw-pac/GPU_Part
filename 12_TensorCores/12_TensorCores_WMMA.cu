/*
 * Tensor Core vs CUDA ALU matrix multiplication exercise with 128-matrix batches per warp and FP32 C accumulator
 *
 * This program computes small matrix multiply-accumulates:
 *
 *     C = A * B + C
 *
 * Matrix dimensions:
 *     A: M x K, FP16
 *     B: K x N, FP16
 *     C: M x N, FP32 input/output accumulator
 *
 * The shape 32x8x16 is a native WMMA tile shape for FP16 tensor-core
 * matrix multiply with FP32 accumulation. Therefore, one warp can compute
 * the whole matrix tile.
 *
 * Three implementations are compared for correctness:
 *     1. Tensor Core implementation using nvcuda::wmma
 *     2. CUDA ALU implementation using one warp, without tensor cores
 *     3. Simple CPU reference implementation
 *
 * In addition, this version benchmarks the Tensor Core and CUDA ALU paths on
 * J different random matrices. Each CUDA block contains one warp, and each warp
 * loops over BATCH_ELEMENTS_PER_WARP consecutive matrices to increase the amount of work performed
 * per launched warp. Therefore, the total batch size must be a multiple of BATCH_ELEMENTS_PER_WARP,
 * and the number of launched warps is J / BATCH_ELEMENTS_PER_WARP. The CUDA event timing includes
 * the complete device-side workflow inside each kernel: global-memory reads,
 * copies/loads into WMMA fragments or scalar registers, arithmetic,
 * and global-memory stores. It intentionally excludes
 * CPU random-number generation, the one-time host-to-device input upload,
 * and the C reset after warmup.
 *
 * Compile example:
 *     nvcc -O3 -arch=sm_70 12_TensorCores_128BatchPerWarp_FloatC.cu -o tensorcore_vs_alu_128_batch_per_warp_float_c
 *
 * Use the SM version of your GPU if it is newer, for example sm_75,
 * sm_80, sm_86, sm_89, or sm_90.
 *
 * Run:
 *     ./tensorcore_vs_alu_128_batch_per_warp_float_c
 */

 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #include <cuda_fp16.h>
 #include <mma.h>
 
 #include <cmath>
 #include <cstdlib>
 #include <iomanip>
 #include <iostream>
 #include <random>
 #include <vector>
 
 using namespace std;
 using namespace nvcuda;
 
 #define M 16
 #define N 16
 #define K 16
 #define WARP_SIZE 32
 
 #define BATCH_ELEMENTS_PER_WARP 512

 #define J 40960
 
 #define SIZE_A (M * K)
 #define SIZE_B (K * N)
 #define SIZE_C (M * N)
 
 static_assert(J % BATCH_ELEMENTS_PER_WARP == 0,
               "J must be a multiple of BATCH_ELEMENTS_PER_WARP.");
 
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
 #error "This exercise requires compilation for sm_70 or newer because it uses WMMA Tensor Cores."
 #endif
 
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
 
 /*
  * Tensor Core implementation for one matrix.
  *
  * One CUDA block contains exactly one warp. The whole warp cooperatively
  * computes the full 32x8 output tile using one WMMA operation.
  *
  * A and B are stored as FP16. C is stored as FP32 and is both the input
  * accumulator and the output matrix. Since the WMMA accumulator fragment is
  * FP32, C can be loaded directly without a half-to-float conversion tile.
  */
 __global__ void matmulTensorCoreKernel(const half* A, const half* B, float* C) {
     // Declare WMMA fragments for one native WMMA tile.
     wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> aFrag;
     wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> bFrag;
     wmma::fragment<wmma::accumulator, M, N, K, float> cFrag;
 
     // Load A, B, and C fragments. Leading dimensions are row-major strides.
     wmma::load_matrix_sync(aFrag, A, K);
     wmma::load_matrix_sync(bFrag, B, N);
     wmma::load_matrix_sync(cFrag, C, N, wmma::mem_row_major);
 
     // Compute C = A * B + C.
     wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
 
     // Store the FP32 accumulator back to C.
     wmma::store_matrix_sync(C, cFrag, N, wmma::mem_row_major);
 }
 
 /*
  * CUDA ALU implementation for one matrix.
  *
  * This also uses one warp, but no WMMA and no Tensor Cores. Since M = 32,
  * all 32 lanes compute one row each. All arithmetic is scalar FP32 arithmetic
  * on the normal CUDA cores / ALUs after converting the FP16 inputs to FP32.
  */
 __global__ void matmulAluWarpKernel(const half* A, const half* B, float* C) {
     int row = threadIdx.x;
     if (row >= M) {
         return;
     }
 
     float acc[N];
 
 #pragma unroll
     for (int col = 0; col < N; ++col) {
         acc[col] = C[row * N + col];
     }
 
 #pragma unroll
     for (int kk = 0; kk < K; ++kk) {
         float a = __half2float(A[row * K + kk]);
 #pragma unroll
         for (int col = 0; col < N; ++col) {
             float b = __half2float(B[kk * N + col]);
             acc[col] += a * b;
         }
     }
 
 #pragma unroll
     for (int col = 0; col < N; ++col) {
         C[row * N + col] = acc[col];
     }
 }
 
 /*
  * Batched Tensor Core implementation.
  *
  * Grid layout:
  *     blockIdx.x identifies a group of 128 consecutive matrices.
  *     threadIdx.x identifies the lane within the one-warp block.
  *
  * Each launched warp loops over 128 batch elements, so the number of launched
  * warps is batchCount / 128 when batchCount is a multiple of 128.
  */
 __global__ void matmulTensorCoreBatchKernel(const half* A, const half* B, float* C,
                                             int batchCount) {
     int firstBatch = blockIdx.x * BATCH_ELEMENTS_PER_WARP;
     if (firstBatch >= batchCount) {
         return;
     }
 
     for (int batchElement = 0; batchElement < BATCH_ELEMENTS_PER_WARP; ++batchElement) {
         int batch = firstBatch + batchElement;
         if (batch >= batchCount) {
             break;
         }

         //TODO
 
     }
 }
 
 /*
  * Batched CUDA ALU implementation.
  *
  * Grid layout is the same as in matmulTensorCoreBatchKernel. The scalar loads
  * into acc, a, and b are part of the timed workflow and represent the copy into
  * scalar registers before the CUDA-core arithmetic executes.
  */
 __global__ void matmulAluWarpBatchKernel(const half* A, const half* B, float* C,
                                          int batchCount) {
     int firstBatch = blockIdx.x * BATCH_ELEMENTS_PER_WARP;
     int row = threadIdx.x;
 
     if (firstBatch >= batchCount || row >= M) {
         return;
     }
 
     for (int batchElement = 0; batchElement < BATCH_ELEMENTS_PER_WARP; ++batchElement) {
         int batch = firstBatch + batchElement;
         if (batch >= batchCount) {
             break;
         }
 
         const half* matrixA = A + static_cast<size_t>(batch) * SIZE_A;
         const half* matrixB = B + static_cast<size_t>(batch) * SIZE_B;
         float* matrixC = C + static_cast<size_t>(batch) * SIZE_C;
 
         float acc[N];
 
 #pragma unroll
         for (int col = 0; col < N; ++col) {
             acc[col] = matrixC[row * N + col];
         }
 
 #pragma unroll
         for (int kk = 0; kk < K; ++kk) {
             float a = __half2float(matrixA[row * K + kk]);
 #pragma unroll
             for (int col = 0; col < N; ++col) {
                 float b = __half2float(matrixB[kk * N + col]);
                 acc[col] += a * b;
             }
         }
 
 #pragma unroll
         for (int col = 0; col < N; ++col) {
             matrixC[row * N + col] = acc[col];
         }
     }
 }
 
 /*
  * Simple CPU reference implementation.
  * This is intentionally straightforward and not optimized.
  */
 void matmulCPU(const half* A, const half* B, const float* C, float* output) {
     for (int row = 0; row < M; ++row) {
         for (int col = 0; col < N; ++col) {
             float sum = C[row * N + col];
 
             for (int kk = 0; kk < K; ++kk) {
                 float a = __half2float(A[row * K + kk]);
                 float b = __half2float(B[kk * N + col]);
                 sum += a * b;
             }
 
             output[row * N + col] = sum;
         }
     }
 }
 
 /*
  * Frobenius norm of the difference between two MxN matrices:
  *     ||X - Y||_F = sqrt(sum_ij (X_ij - Y_ij)^2)
  */
 double frobeniusNormDiff(const float* reference, const float* result, int size) {
     double sumSquares = 0.0;
 
     for (int i = 0; i < size; ++i) {
         double diff = static_cast<double>(reference[i]) - static_cast<double>(result[i]);
         sumSquares += diff * diff;
     }
 
     return sqrt(sumSquares);
 }
 
 double frobeniusNorm(const float* matrix, int size) {
     double sumSquares = 0.0;
 
     for (int i = 0; i < size; ++i) {
         double value = static_cast<double>(matrix[i]);
         sumSquares += value * value;
     }
 
     return sqrt(sumSquares);
 }
 
 void printComparison(const char* name, const float* reference, const float* result, int size) {
     double diffNorm = frobeniusNormDiff(reference, result, size);
     double refNorm = frobeniusNorm(reference, size);
     double relative = diffNorm / (refNorm + 1e-20);
 
     cout << name << endl;
     cout << "  Frobenius norm ||GPU - CPU||_F: " << diffNorm << endl;
     cout << "  Relative Frobenius error:       " << relative << endl;
 }
 
 void fillRandomBatch(vector<half>& hostA, vector<half>& hostB, vector<float>& hostC) {
     mt19937 rng(1234);
     uniform_real_distribution<float> dist(-1.0f, 1.0f);
 
     for (int batch = 0; batch < J; ++batch) {
         size_t baseA = static_cast<size_t>(batch) * SIZE_A;
         size_t baseB = static_cast<size_t>(batch) * SIZE_B;
         size_t baseC = static_cast<size_t>(batch) * SIZE_C;
 
         for (int i = 0; i < SIZE_A; ++i) {
             hostA[baseA + i] = __float2half(dist(rng));
         }
         for (int i = 0; i < SIZE_B; ++i) {
             hostB[baseB + i] = __float2half(dist(rng));
         }
         for (int i = 0; i < SIZE_C; ++i) {
             hostC[baseC + i] = dist(rng);
         }
 
         /*
          * Make the per-matrix uniqueness explicit and deterministic.
          *
          * J = 10240 = 80 x 128. These two entries encode the batch id as
          * a pair of values with large spacing, so no two matrices in the batch
          * share the same pair after FP16 conversion.
          */
         int low = batch % BATCH_ELEMENTS_PER_WARP;
         int high = batch / BATCH_ELEMENTS_PER_WARP;
         int warpBatches = J / BATCH_ELEMENTS_PER_WARP;
         float lowValue = -1.0f + 2.0f * static_cast<float>(low) /
                          static_cast<float>(BATCH_ELEMENTS_PER_WARP - 1);
         float highValue = (warpBatches > 1)
                               ? -1.0f + 2.0f * static_cast<float>(high) /
                                             static_cast<float>(warpBatches - 1)
                               : 0.0f;
         hostA[baseA + 0] = __float2half(lowValue);
         hostA[baseA + 1] = __float2half(highValue);
     }
 }
 
 float timeTensorCoreBatch(const half* deviceA, const half* deviceB, float* deviceC,
                           int batchCount) {
    // TODO
 }
 
 float timeAluBatch(const half* deviceA, const half* deviceB, float* deviceC,
                    int batchCount) {
     cudaEvent_t start;
     cudaEvent_t stop;
     gpuErrCheck(cudaEventCreate(&start));
     gpuErrCheck(cudaEventCreate(&stop));
 
     int warpCount = batchCount / BATCH_ELEMENTS_PER_WARP;
 
     gpuErrCheck(cudaEventRecord(start));
     matmulAluWarpBatchKernel<<<warpCount, WARP_SIZE>>>(deviceA, deviceB, deviceC,
                                                        batchCount);
     gpuErrCheck(cudaPeekAtLastError());
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
 
     if (prop.major < 7) {
         cout << "This exercise needs Tensor Core WMMA support, i.e. compute capability 7.0 or newer." << endl;
         return 0;
     }
 
     // Allocate and initialize J distinct random matrices on the host.
     vector<half> hostA(static_cast<size_t>(J) * SIZE_A);
     vector<half> hostB(static_cast<size_t>(J) * SIZE_B);
     vector<float> hostC(static_cast<size_t>(J) * SIZE_C);
     fillRandomBatch(hostA, hostB, hostC);
 
     vector<float> hostOutputCPU(SIZE_C);
     vector<float> hostOutputTensorCore(SIZE_C);
     vector<float> hostOutputALU(SIZE_C);
 
     // Use matrix 0 for the correctness check.
     const int checkBatch = 0;
     const half* checkA = hostA.data() + static_cast<size_t>(checkBatch) * SIZE_A;
     const half* checkB = hostB.data() + static_cast<size_t>(checkBatch) * SIZE_B;
     const float* checkC = hostC.data() + static_cast<size_t>(checkBatch) * SIZE_C;
     matmulCPU(checkA, checkB, checkC, hostOutputCPU.data());
 
     // Allocate device memory for the full batch.
     half* deviceA;
     half* deviceB;
     float* deviceCTensorCore;
     float* deviceCALU;
 
     size_t bytesA = static_cast<size_t>(J) * SIZE_A * sizeof(half);
     size_t bytesB = static_cast<size_t>(J) * SIZE_B * sizeof(half);
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
 
     int launchedWarps = J / BATCH_ELEMENTS_PER_WARP;
 
     // Warm up both paths once so the benchmark does not include one-time setup effects.
     matmulTensorCoreBatchKernel<<<launchedWarps, WARP_SIZE>>>(deviceA, deviceB, deviceCTensorCore,
                                                              J);
     gpuErrCheck(cudaPeekAtLastError());
     matmulAluWarpBatchKernel<<<launchedWarps, WARP_SIZE>>>(deviceA, deviceB, deviceCALU,
                                                            J);
     gpuErrCheck(cudaPeekAtLastError());
     gpuErrCheck(cudaDeviceSynchronize());
 
     // The kernels update C in place. Reset C after warmup so the timed run
     // starts from the same initial accumulator values as the CPU reference.
     gpuErrCheck(cudaMemcpy(deviceCTensorCore, hostC.data(), bytesC, cudaMemcpyHostToDevice));
     gpuErrCheck(cudaMemcpy(deviceCALU, hostC.data(), bytesC, cudaMemcpyHostToDevice));
 
     // Timed benchmark over J different matrices.
     float tensorCoreMs = timeTensorCoreBatch(deviceA, deviceB, deviceCTensorCore, J);
     float aluMs = timeAluBatch(deviceA, deviceB, deviceCALU, J);
 
     // Copy the result for one matrix back to the CPU for correctness reporting.
     gpuErrCheck(cudaMemcpy(hostOutputTensorCore.data(),
                            deviceCTensorCore + static_cast<size_t>(checkBatch) * SIZE_C,
                            SIZE_C * sizeof(float), cudaMemcpyDeviceToHost));
     gpuErrCheck(cudaMemcpy(hostOutputALU.data(),
                            deviceCALU + static_cast<size_t>(checkBatch) * SIZE_C,
                            SIZE_C * sizeof(float), cudaMemcpyDeviceToHost));
 
     cout << fixed << setprecision(10) << endl;
     printComparison("Tensor Core WMMA result compared with CPU reference:",
                     hostOutputCPU.data(), hostOutputTensorCore.data(), SIZE_C);
     printComparison("CUDA ALU warp result compared with CPU reference:",
                     hostOutputCPU.data(), hostOutputALU.data(), SIZE_C);
 
     cout << endl << "N=" << N << " M=" << M << " K=" << K << " J=" << J << endl;
     cout << "Matrices per launched warp: " << BATCH_ELEMENTS_PER_WARP << endl;
     cout << "Launched warps per path:    " << launchedWarps << endl;
 
     // Show a few values so students can see that the matrices agree element-wise.
     cout << endl << "First 8 C/output elements for checked matrix " << checkBatch << ":" << endl;
     cout << "  CPU:         ";
     for (int i = 0; i < 8; ++i) cout << hostOutputCPU[i] << " ";
     cout << endl;
 
     cout << "  Tensor Core: ";
     for (int i = 0; i < 8; ++i) cout << hostOutputTensorCore[i] << " ";
     cout << endl;
 
     cout << "  CUDA ALU:    ";
     for (int i = 0; i < 8; ++i) cout << hostOutputALU[i] << " ";
     cout << endl;
 
     double tensorCoreAvgUs = static_cast<double>(tensorCoreMs) * 1000.0 / static_cast<double>(J);
     double aluAvgUs = static_cast<double>(aluMs) * 1000.0 / static_cast<double>(J);
     double speedup = aluAvgUs / (tensorCoreAvgUs + 1e-20);
 
     cout << endl << "Execution-time benchmark over " << J << " different random matrices:" << endl;
     cout << "  Timing scope: one batched kernel per path; includes device global-memory loads," << endl;
     cout << "                copies/loads into WMMA fragments or scalar registers, arithmetic," << endl;
     cout << "                and global-memory stores." << endl;
     cout << "  Not timed:    CPU random-number generation, one-time host-to-device input upload, and C reset after warmup." << endl;
     cout << endl;
     cout << "  Tensor Core total time:     " << tensorCoreMs << " ms" << endl;
     cout << "  CUDA ALU total time:        " << aluMs << " ms" << endl;
     cout << "  Tensor Core average/matrix: " << tensorCoreAvgUs << " us" << endl;
     cout << "  CUDA ALU average/matrix:    " << aluAvgUs << " us" << endl;
     cout << "  ALU / Tensor Core time:     " << speedup << "x" << endl;
 
     if (speedup > 1.0) {
         cout << "  Tensor Core is faster by:   " << speedup << "x" << endl;
     } else {
         cout << "  CUDA ALU is faster by:      " << (1.0 / speedup) << "x" << endl;
     }
 
     // Free device memory.
     gpuErrCheck(cudaFree(deviceA));
     gpuErrCheck(cudaFree(deviceB));
     gpuErrCheck(cudaFree(deviceCTensorCore));
     gpuErrCheck(cudaFree(deviceCALU));
 
     return 0;
 }
 