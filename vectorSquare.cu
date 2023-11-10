/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Some modification for UCA Lab on october 2021
 */


#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector multiplication of A into C.
 * The 3 vectors have the same number of elements numElements.
 */
__global__
void vectorAdd(const float *A, float *C, unsigned long numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * A[i];
    }
}

void checkErr(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s (error code %d: '%s')!\n", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
    // Get the property device
    // TO BE COMPLETED
    int threadsPerBlock=1024;
    int maxBlocks=2147483647;
    printf("max of %d blocks of %d threads\n", maxBlocks, threadsPerBlock);
    
    // To mesure different execution time
    cudaEvent_t start_copying_to_device, stop_copying_to_device;
    cudaEvent_t start_sum, stop_sum;
    cudaEvent_t start_copying_to_host, stop_copying_to_host;
    cudaEvent_t start_seq, stop_seq;
    
    float CUDA1, CUDA2, CUDA3, SEQ;

    cudaEventCreate(&start_copying_to_device); 
    cudaEventCreate(&stop_copying_to_device);

    cudaEventCreate(&start_sum); 
    cudaEventCreate(&stop_sum);

    cudaEventCreate(&start_copying_to_host); 
    cudaEventCreate(&stop_copying_to_host);
    
    cudaEventCreate(&start_seq); 
    cudaEventCreate(&stop_seq);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    unsigned long numElements = 50000;
    if (argc == 2) {
      numElements = strtoul( argv[1] , 0, 10 );
    }
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %lu elements]\n", numElements);

    // Allocate the host input vectors A
    float * h_A = (float *)malloc(size);
    // Allocate the host output vector C
    float * h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
    }

    // 1a. Allocate the device input vectors A

    float * d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    checkErr(err, "Failed to allocate device vector A");

    // 1.b. Allocate the device output vector C
    float * d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    checkErr(err, "Failed to allocate device vector C");

    // 2. Copy the host input vectors A
    //     to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");

    cudaEventCreate(&start_copying_to_device); 
    cudaEventCreate(&stop_copying_to_device);
    cudaEventRecord(start_copying_to_device, 0);

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy device vector A from host to device");

    cudaEventRecord(stop_copying_to_device, 0);
    cudaEventSynchronize(stop_copying_to_device);
    cudaEventElapsedTime(&CUDA1, start_copying_to_device, stop_copying_to_device);
    cudaEventDestroy(start_copying_to_device);
    cudaEventDestroy(stop_copying_to_device);

    printf("CUDA copying time from host to device: %lf\n", CUDA1); 

    // 3. Launch the Vector Add CUDA Kernel

    cudaEventCreate(&start_sum); 
    cudaEventCreate(&stop_sum);
    cudaEventRecord(start_sum, 0);

    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid>maxBlocks)
    {
        fprintf(stderr, "too much blocks %d!\n", blocksPerGrid);
        exit(EXIT_FAILURE);
    } else
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements);
    err = cudaGetLastError();
    checkErr(err, "Failed to launch vectorA:dd kernel");
    
    cudaEventRecord(stop_sum, 0);
    cudaEventSynchronize(stop_sum);
    cudaEventElapsedTime(&CUDA2, start_sum, stop_sum);
    cudaEventDestroy(start_sum);
    cudaEventDestroy(stop_sum);
    
    printf("CUDA executing time on device: %lf\n", CUDA2); 

    // 4. Copy the device result vector in device memory
    //     to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaEventCreate(&start_copying_to_host); 
    cudaEventCreate(&stop_copying_to_host);
    cudaEventRecord(start_copying_to_host, 0);
    
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy vector C from device to host");
    
    cudaEventRecord(stop_copying_to_host, 0);
    cudaEventSynchronize(stop_copying_to_host);
    cudaEventElapsedTime(&CUDA3, start_copying_to_host, stop_copying_to_host);
    cudaEventDestroy(start_copying_to_host);
    cudaEventDestroy(stop_copying_to_host);

    printf("CUDA copying time from device to host: %lf\n", CUDA3); 

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] * h_A[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("CUDA test PASSED\n");
    printf("CUDA time: %lf\n", CUDA1+CUDA2+CUDA3); 
    printf("CUDA blocksPerGrid= %i, threadsPerBlock= %i\n", blocksPerGrid, threadsPerBlock);

    // Free device global memory
    err = cudaFree(d_A);
    checkErr(err, "Failed to free device vector A");

    err = cudaFree(d_C);
    checkErr(err, "Failed to free device vector C");

    // repeat the computation sequentially
    cudaEventCreate(&start_seq); 
    cudaEventCreate(&stop_seq);
    cudaEventRecord(start_seq, 0);

    for (int i = 0; i < numElements; ++i)
    {
       h_C[i] = h_A[i] * h_A[i];
    }
    
    cudaEventRecord(stop_seq, 0);
    cudaEventSynchronize(stop_seq);
    cudaEventElapsedTime(&SEQ, start_seq, stop_seq);
    cudaEventDestroy(start_seq);
    cudaEventDestroy(stop_seq);

    // verify again
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] * h_A[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("\nNormal test PASSED\n");
    
    printf("Normal time: %lf\n", SEQ); 
    
    // Free host memory
    free(h_A);
    free(h_C);

    // Reset the device and exit
    err = cudaDeviceReset();
    checkErr(err, "Unable to reset device");
    fprintf(stderr, "%d, %lf, %lf, %lf, %lf\n", numElements, CUDA1, CUDA2, CUDA3, SEQ);

    printf("Done\n");
    return 0;
}

