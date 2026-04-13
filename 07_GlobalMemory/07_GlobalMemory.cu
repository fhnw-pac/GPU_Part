/*
* This program will do a load and store operation on each element of a vector.
* The access to the vector is strided (where stride 1 = coalesced).
* It meassures the bandwidth in GB/s for different stride sizes and on
* CPU and GPU.
*
* +---------+                        +---------+
* |111111111| + 1 (on each Thread) = |222222222|
* +---------+                        +---------+
*
* vector   = all Ones
*
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <chrono> 

using namespace std;


/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


// GPU kernel which access an vector with a strdie pattern
__global__ void strided_kernel(int* vec, int size, int stride)
{
    //ToDo: Implement the strided kernel vec[i] = vec[i] + 1 
}


// Execute a loop of different strides accessing a vector as GPU kernels.
// Meassure the spent time and print out the reached bandwidth in GB/s.
void gpu_stride_loop(int* device_vec, int size)
{
    // Define some helper values
    const int processedMB = size * sizeof(int) / 1024 / 1024 * 2;  // 2x as 1 read and 1 write
    const int blockSize = 256;
    float ms;

    // Init CUDA events used to meassure timings 
    cudaEvent_t startEvent, stopEvent;
    gpuErrCheck(cudaEventCreate(&startEvent));
    gpuErrCheck(cudaEventCreate(&stopEvent));

    // Warm up GPU (The first kernel of a program has more overhead than the followings)
    gpuErrCheck(cudaEventRecord(startEvent, 0));
    strided_kernel << <size / blockSize, blockSize >> > (device_vec, size, 1);
    gpuErrCheck(cudaEventRecord(stopEvent, 0));
    gpuErrCheck(cudaEventSynchronize(stopEvent));

    gpuErrCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    cout << "GPU warmup kernel: " << processedMB / ms << "GB/s bandwidth" << endl;

    // ToDo: Implement the strided loop analogue the CPU implementation
    //       Calculate and print the used Bandwidth
    //       No need to reset the device_vec to 1, we are not interessted in the result

}


// Execute a loop of different strides accessing a vector.
// Meassure the spent time and print out the reached bandwidth in GB/s.
void cpu_stride_loop(int* vec, int size)
{
    float processedMB = size * sizeof(int) / 1024 / 1024 * 2;  // 2x as 1 read and 1 write
    for (int stride = 1; stride <= 32; stride++) {
        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < size; i++) {
            int strided_i = (i * stride);
            strided_i = strided_i  % size + (strided_i / size);
            vec[strided_i] = vec[strided_i] + 1;
        }

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "CPU stride size " << stride << ": " << processedMB / duration.count() << "GB/s bandwidth" << endl;
    }
}


// Init und destruct memory and call the CPU and the GPU meassurment code.
int main(void)
{
    // Define the size of the vector in MB
    const int width_MB = 128;
    const int width = width_MB * 1024 * 1024 / sizeof(int);

    // Allocate and prepare input vector
    int* hostVector = new int[width];
    for (int index = 0; index < width; index++) {
        hostVector[index] = 1;
    }

    // Allocate device memory
    int* deviceVector;
    gpuErrCheck(cudaMalloc(&deviceVector, width * sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVector, hostVector, width * sizeof(int), cudaMemcpyHostToDevice));

    // run stride loop on CPU to have some reference values
    cpu_stride_loop(hostVector, width);
    cout << "--------------------------------------------------------" << endl;

    // run stride loop on GPU
    gpu_stride_loop(deviceVector, width);

    // Free memory on device & host
    cudaFree(deviceVector);
    delete[] hostVector;

    return 0;
}