
#include <stdio.h>
#include<stdint.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include<chrono>
int main(int argc, const char* argv[])
{
    uint32_t **data=(uint32_t **)malloc(512*8*sizeof(uint32_t *));
    for(int i=0;i<512*8;i++)
    {
        cudaHostAlloc((void **)&data[i],4096*sizeof(uint32_t),cudaHostAllocMapped);
    }
    for(int i=0;i<512*8;i++)
    {
        for(int j=0;j<4096;j++)
        {
            data[i][j]=i*4096+j;
        }
    }
    uint32_t *d_data;
    cudaMalloc((void **)&d_data,512*4096*8*sizeof(uint32_t));
    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<512*8;i++)
    {
        cudaMemcpy(d_data+i*4096,data[i],4096*sizeof(uint32_t),cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    printf("elapsed time: %f s\n", elapsed_seconds.count());

    uint32_t *h_data;
    cudaHostAlloc((void **)&h_data,512*4096*8*sizeof(uint32_t),cudaHostAllocMapped);
    cudaDeviceSynchronize();

    start = std::chrono::steady_clock::now();
    for(int i=0;i<512*8;i++)
    {
        for(int j=0;j<4096;j++)
        {
            
            h_data[i*4096+j]=data[i][j];
        }
    }
    cudaMemcpy(d_data,h_data,512*4096*8*sizeof(uint32_t),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    printf("elapsed time: %f s\n", elapsed_seconds.count());
    return 0;
}