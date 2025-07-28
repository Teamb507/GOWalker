#pragma once
#include <stdio.h>
#include "engine/cache.hpp"
#include "engine/walk.hpp"
#include <curand.h>
#include <engine/cache.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "api/types.hpp"
// gpu初始化
gpu_walks* initwalk(wid_t nwalks, int hops, bid_t totblocks, cudaStream_t stream)
{
   gpu_walks* m = (gpu_walks*)malloc(sizeof(gpu_walks));
   gpu_walks* g_walks;
   checkCudaError(cudaMallocAsync((void**)&(m->walk_offset), sizeof(wid_t) * nwalks, stream));
   checkCudaError(cudaMallocAsync((void**)&(m->block_offset), sizeof(wid_t) * (totblocks + 1), stream));
   checkCudaError(cudaMemsetAsync(m->block_offset, 0, sizeof(wid_t) * (totblocks + 1), stream));
   checkCudaError(cudaMemsetAsync(m->walk_offset, -1, sizeof(wid_t) * nwalks, stream));
   checkCudaError(cudaMallocAsync((void**)&(m->walks), sizeof(walker_t) * nwalks, stream));
   checkCudaError(cudaMallocAsync((void**)&(m->res_walks), sizeof(walker_t) * nwalks, stream));
   checkCudaError(cudaMallocAsync((void**)&(g_walks), sizeof(gpu_walks), stream));
   m->hops = hops; // 需要更改
   m->nwalk = 0;
   checkCudaError(cudaMemcpyAsync(g_walks, m, sizeof(gpu_walks), cudaMemcpyHostToDevice, stream));
   return g_walks;
}

gpu_graph* initgpugraph(graph_block* cpu_b, cudaStream_t stream) // 初始化gpu内存中的图存储区
{
   gpu_graph* g_graph;
   gpu_graph* m = (gpu_graph*)malloc(sizeof(gpu_graph));
   gpu_graph_block* m_b = (gpu_graph_block*)malloc(sizeof(gpu_graph_block) * cpu_b->nblocks);
   for (int i = 0; i < cpu_b->nblocks; i++)
   {
      m_b[i].blk = cpu_b->blocks[i].blk;
      m_b[i].index = cpu_b->blocks[i].cache_index;
      m_b[i].nedges = cpu_b->blocks[i].nedges;
      m_b[i].nverts = cpu_b->blocks[i].nverts;
      m_b[i].start_vert = cpu_b->blocks[i].start_vert;
      m_b[i].start_edge = cpu_b->blocks[i].start_edge;
      m_b[i].status = cpu_b->blocks[i].status;
   }
   m->nblock = cpu_b->nblocks;
   checkCudaError(cudaMallocAsync((void**)&(m->blocks), sizeof(gpu_graph_block) * cpu_b->nblocks, stream));
   checkCudaError(cudaMemcpyAsync(m->blocks, m_b, sizeof(gpu_graph_block) * cpu_b->nblocks, cudaMemcpyHostToDevice, stream));
   checkCudaError(cudaMallocAsync((void**)&g_graph, sizeof(gpu_graph), stream));
   checkCudaError(cudaMemcpyAsync(g_graph, m, sizeof(gpu_graph), cudaMemcpyHostToDevice, stream));
   return g_graph;
}

void initcacheblock(gpu_block* gpu, graph_cache* cpu, bool weight, size_t versize,size_t edgesize, cudaStream_t stream)
{
   for (int i = 0; i < cpu->ncblock; i++)
   {
      checkCudaError(cudaMallocAsync((void**)&(gpu[i].beg_pos), versize, stream));
      checkCudaError(cudaMallocAsync((void**)&(gpu[i].csr), edgesize, stream));
      if (weight)
      {
         checkCudaError(cudaMallocAsync((void**)&(gpu[i].weights), edgesize, stream));
      }
      gpu[i].blk = 0;
   }
}

gpu_cache* initgpucache(graph_cache *allcache,graph_block* cpu_b, graph_cache* cpu_c, bool weight, size_t blocksize, cudaStream_t stream)
{
   eid_t nedges = cpu_b->blocks[0].nedges;
   bid_t ncblock = cpu_c->ncblock;
   bid_t nblock = cpu_b->nblocks;
   eid_t maxver = 0;
   eid_t maxedge = 0;
   for (int i = 0; i < cpu_b->nblocks; i++) {
   //   std::cout<<"blk:"<<i<<"   vers:"<<cpu_b->blocks[i].nverts<<",  nedges:"<<cpu_b->blocks[i].nedges<<", csr:["<<allcache->cache_blocks[i].beg_pos[0]<<"]"<<std::endl;
      if (cpu_b->blocks[i].nverts > maxver)
         maxver = cpu_b->blocks[i].nverts;
      if (cpu_b->blocks[i].nedges > maxedge)
         maxedge = cpu_b->blocks[i].nedges;
   }
   // std::cout<<"blocksize:"<<blocksize<<" ,  versize:"<<sizeof(eid_t) * (maxver + 1)<<" , edgesize:"<<sizeof(vid_t) * (maxedge)<<std::endl;
   size_t versize=blocksize<(sizeof(eid_t) * (maxver + 1))?blocksize:(sizeof(eid_t) * (maxver + 1));
   size_t edgesize=blocksize<(sizeof(vid_t) * (maxedge))?blocksize:(sizeof(vid_t) * (maxedge ));
   maxedge=edgesize/sizeof(vid_t);
   for (int i = 0; i < cpu_b->nblocks; i++) {
     if (cpu_b->blocks[i].nedges > maxedge) {
       cpu_b->blocks[i].nedges = maxedge;
       if (cpu_b->blocks[i].nverts == 1) {
         allcache->cache_blocks[i].beg_pos[0]=allcache->cache_blocks[i-1].beg_pos[allcache->cache_blocks[i-1].block->nverts];
         allcache->cache_blocks[i].beg_pos[1]=allcache->cache_blocks[i].beg_pos[0]+maxedge;
       } else {
         std::cout<<"graph convert false!"<<std::endl;
       }
       }
   }
   // for (int i = 0; i < cpu_b->nblocks; i++) {
   //    std::cout<<"blk:"<<i<<"   vers:"<<cpu_b->blocks[i].nverts<<",  nedges:"<<cpu_b->blocks[i].nedges<<", csr:["<<allcache->cache_blocks[i].beg_pos[0]<<"]"<<std::endl;
   //  }
   gpu_cache* g_cache;
   gpu_cache* m_c = (gpu_cache*)malloc(sizeof(gpu_cache));
   gpu_block* m_b = (gpu_block*)malloc(sizeof(gpu_block) * ncblock);
   initcacheblock(m_b, cpu_c, weight, versize,edgesize, stream);
   m_c->ncblock = ncblock;
   checkCudaError(cudaMallocAsync((void**)&(m_c->cache_blocks), sizeof(gpu_block) * ncblock, stream));
   checkCudaError(cudaMemcpyAsync(m_c->cache_blocks, m_b, sizeof(gpu_block) * ncblock, cudaMemcpyHostToDevice, stream));
   checkCudaError(cudaMallocAsync((void**)&(g_cache), sizeof(gpu_cache), stream));
   checkCudaError(cudaMemcpyAsync(g_cache, m_c, sizeof(gpu_cache), cudaMemcpyHostToDevice, stream));
   return g_cache;
}

__global__ void initrand(curandState* states, uint64_t seed)
{
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   curand_init(seed, tid, 0, &states[tid]);
}
curandState* cudarand(int blockdim, int threadperblock, cudaStream_t stream)
{
   curandState* states;
   std::chrono::nanoseconds time;
   auto seed = time.count();
   checkCudaError(cudaMallocAsync(&states, sizeof(curandState) * blockdim * threadperblock, stream));
   initrand << <blockdim, threadperblock, 0, stream >> > (states, seed);
   return states;
}
// gpu调度的传输
void copycachetogpu(gpu_cache* g_cache, graph_cache* c_cache, bid_t index, bool weight, cudaStream_t copy)
{

   gpu_cache* m = (gpu_cache*)malloc(sizeof(gpu_cache));
   checkCudaError(cudaMemcpyAsync(m, g_cache, sizeof(gpu_cache), cudaMemcpyDeviceToHost, copy));
   gpu_block* m_b = (gpu_block*)malloc(sizeof(gpu_block));
   checkCudaError(cudaMemcpyAsync(m_b, &(m->cache_blocks[index]), sizeof(gpu_block), cudaMemcpyDeviceToHost, copy));
   m_b->blk = c_cache->cache_blocks[index].block->blk;
   checkCudaError(cudaMemcpyAsync(m_b->beg_pos, c_cache->cache_blocks[index].beg_pos, sizeof(eid_t) * (c_cache->cache_blocks[index].block->nverts + 1), cudaMemcpyHostToDevice, copy));
   checkCudaError(cudaMemcpyAsync(m_b->csr, c_cache->cache_blocks[index].csr, sizeof(vid_t) * c_cache->cache_blocks[index].block->nedges, cudaMemcpyHostToDevice, copy));
   if (weight)
   {
      checkCudaError(cudaMemcpyAsync(m_b->weights, c_cache->cache_blocks[index].weights, sizeof(real_t) * c_cache->cache_blocks[index].block->nedges, cudaMemcpyHostToDevice, copy));
   }
   checkCudaError(cudaMemcpyAsync(&(m->cache_blocks[index]), m_b, sizeof(gpu_block), cudaMemcpyHostToDevice, copy));
}

void copygraphtogpu(gpu_graph* g_graph, graph_block* cpu_b)
{
   gpu_graph* m = (gpu_graph*)malloc(sizeof(gpu_graph));
   checkCudaError(cudaMemcpy(m, g_graph, sizeof(gpu_graph), cudaMemcpyDeviceToHost));
   gpu_graph_block* m_b = (gpu_graph_block*)malloc(sizeof(gpu_graph_block) * cpu_b->nblocks);
   for (int i = 0; i < cpu_b->nblocks; i++)
   {
      m_b[i].blk = cpu_b->blocks[i].blk;
      m_b[i].index = cpu_b->blocks[i].cache_index;
      m_b[i].nedges = cpu_b->blocks[i].nedges;
      m_b[i].nverts = cpu_b->blocks[i].nverts;
      m_b[i].start_vert = cpu_b->blocks[i].start_vert;
      m_b[i].start_edge = cpu_b->blocks[i].start_edge;
      m_b[i].status = cpu_b->blocks[i].status;
   }
   checkCudaError(cudaMemcpy(m->blocks, m_b, sizeof(gpu_graph_block) * cpu_b->nblocks, cudaMemcpyHostToDevice));
   checkCudaError(cudaMemcpy(g_graph, m, sizeof(gpu_graph), cudaMemcpyHostToDevice));
}

void copywalktogpu(gpu_walks* gpu, walker_t* cpu, wid_t nwalks, int hops, int offset, bid_t nblocks, cudaStream_t stream)
{
   assert(gpu != NULL);
   gpu_walks* m = (gpu_walks*)malloc(sizeof(gpu_walks));
   checkCudaError(cudaMemcpyAsync(m, gpu, sizeof(gpu_walks), cudaMemcpyDeviceToHost, stream));
   m->hops = hops;
   if (offset == 0)
   {
      m->nwalk = 0;
      m->res_nwalk = 0;
      checkCudaError(cudaMemsetAsync(m->block_offset, 0, sizeof(wid_t) * (nblocks * nblocks + 1), stream));
   }
   m->nwalk += nwalks;
   checkCudaError(cudaMemsetAsync(&(m->walk_offset[offset]), -1, sizeof(wid_t) * nwalks, stream));
   checkCudaError(cudaMemcpyAsync(&(m->walks[offset]), cpu, sizeof(walker_t) * nwalks, cudaMemcpyHostToDevice, stream));
   checkCudaError(cudaMemcpyAsync(gpu, m, sizeof(gpu_walks), cudaMemcpyHostToDevice, stream));
}

// gpu释放分配内存
void freewalk(gpu_walks* g_walks)
{
   gpu_walks m;
   checkCudaError(cudaMemcpy(&m, g_walks, sizeof(gpu_walks), cudaMemcpyDeviceToHost));
   checkCudaError(cudaFree(m.walks));
   checkCudaError(cudaFree(m.res_walks));
   checkCudaError(cudaFree(m.walk_offset));
   checkCudaError(cudaFree(m.block_offset));
   checkCudaError(cudaFree(g_walks));
}
void freecache(gpu_cache* g_cache)
{
   gpu_cache m;
   checkCudaError(cudaMemcpy(&m, g_cache, sizeof(gpu_cache), cudaMemcpyDeviceToHost));
   checkCudaError(cudaFree(m.cache_blocks));
   checkCudaError(cudaFree(g_cache));
}
void freegraph(gpu_graph* g_graph)
{
   gpu_graph m;
   checkCudaError(cudaMemcpy(&m, g_graph, sizeof(gpu_graph), cudaMemcpyDeviceToHost));
   checkCudaError(cudaFree(m.blocks));
   checkCudaError(cudaFree(g_graph));
}

void freerand(curandState* states)
{
   checkCudaError(cudaFree(states));
}

void pipeline_gpugraph(gpu_graph* g_graph, block_t* b, cudaStream_t copy) // 初始化gpu内存中的图存储区
{
   gpu_graph* m = (gpu_graph*)malloc(sizeof(gpu_graph));
   checkCudaError(cudaMemcpyAsync(m, g_graph, sizeof(gpu_graph), cudaMemcpyDeviceToHost, copy));
   gpu_graph_block* m_b = (gpu_graph_block*)malloc(sizeof(gpu_graph_block));
   bid_t blk = b->blk;
   m_b->blk = b->blk;
   m_b->index = b->cache_index;
   m_b->nedges = b->nedges;
   m_b->nverts = b->nverts;
   m_b->start_vert = b->start_vert;
   m_b->start_edge = b->start_edge;
   m_b->status = b->status;
   checkCudaError(cudaMemcpyAsync(&(m->blocks[blk]), m_b, sizeof(gpu_graph_block), cudaMemcpyHostToDevice, copy));
   return;
}

