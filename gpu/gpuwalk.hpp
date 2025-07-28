#pragma once
#include "memory.hpp"
#include "api/types.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <nvToolsExt.h>
__device__ double g_max(double a, double b)
{
    if (a >= b)
        return a;
    return b;
}
__device__ double g_min(double a, double b)
{
    if (a <= b)
        return a;
    return b;
}

__device__ bool search(vid_t *begin, vid_t *end, vid_t v, int nedges)
{
    // for(int i=0;begin[i]!=*end;i++)
    // {
    //     if(begin[i]==v)
    //         return true;
    // }
    // return false;
    int head = 0;
    int tail = nedges - 1;
    int i = 0;
    while (head <= tail && (begin[i] <= *end))
    {
        i = (head + tail) / 2;
        if (v > begin[i])
        {
            head = i + 1;
        }
        else if (v < begin[i])
        {
            tail = i - 1;
        }
        else
        {
            return true;
        }
    }
    return false;
}

__device__ bid_t getindex(gpu_graph *g_graph, gpu_cache *g_cache, vid_t ver)
{
    for (int i = 0; i < g_graph->nblock; i++)
    {
        if (ver >= g_graph->blocks[i].start_vert && ver < (g_graph->blocks[i].start_vert + g_graph->blocks[i].nverts))
        {
            if (g_graph->blocks[i].status == ACTIVE)
            {
                return g_graph->blocks[i].index;
            }
            else
            {
                return g_graph->nblock;
            }
        }
    }
    return g_graph->nblock;
}

__device__ bid_t getblk(gpu_graph *g_graph, vid_t ver)
{
    for (int i = 0; i < g_graph->nblock; i++)
    {
        if (ver >= g_graph->blocks[i].start_vert && ver < (g_graph->blocks[i].start_vert + g_graph->blocks[i].nverts))
            return g_graph->blocks[i].blk;
    }
}

__device__ bid_t zerocopy_getblk(bid_t nblocks, vid_t ver, gpu_graph *g_graph)
{
    for (int i = 0; i < nblocks; i++)
    {
        if (ver >= g_graph->blocks[i].start_vert && ver < (g_graph->blocks[i].start_vert + g_graph->blocks[i].nverts))
            return g_graph->blocks[i].blk;
    }
}

__device__ void gpu_update(wid_t index, gpu_walks *g_walks, gpu_cache *g_cache, gpu_graph *g_graph, real_t p, real_t q, hid_t maxhop, bid_t ncblock, curandState state, gpu_test *test)
{
    vid_t cur_vertex = g_walks->walks[index].current;
    vid_t prev_vertex = g_walks->walks[index].previous;
    bid_t cur_blk = g_walks->walks[index].cur_index;
    bid_t prev_blk = g_walks->walks[index].prev_index;
    hid_t hop = g_walks->walks[index].hop;
    bid_t cur_index = g_graph->blocks[cur_blk].index;
    bid_t prev_index = g_graph->blocks[prev_blk].index;
    if (cur_index == ncblock || prev_index == ncblock)
        return;
    wid_t run_step = 0;

    while (cur_index != ncblock && hop < maxhop)
    {
        vid_t start_ver = g_graph->blocks[cur_blk].start_vert;
        vid_t prev_start_ver = g_graph->blocks[prev_blk].start_vert;
        vid_t off = cur_vertex - start_ver;
        vid_t prev_off = prev_vertex - prev_start_ver;
        eid_t adj_head = g_cache->cache_blocks[cur_index].beg_pos[off] - g_graph->blocks[cur_blk].start_edge;
        eid_t adj_tail = g_cache->cache_blocks[cur_index].beg_pos[off + 1] - g_graph->blocks[cur_blk].start_edge;
        eid_t prev_adj_head = g_cache->cache_blocks[prev_index].beg_pos[prev_off] - g_graph->blocks[prev_blk].start_edge;
        eid_t prev_adj_tail = g_cache->cache_blocks[prev_index].beg_pos[prev_off + 1] - g_graph->blocks[prev_blk].start_edge;
        vid_t next_vertex = 0;
        eid_t deg = adj_tail - adj_head;
        if (deg == 0)
        {
            hop = maxhop - 1;
        }
        else
        {
            // if(deg>=g_graph->blocks[cur_blk].nedges) return;
            real_t max_val = g_max(1.0 / p, g_max(1.0 / q, 1.0));
            real_t min_val = g_min(1.0 / p, g_min(1.0 / q, 1.0));
            bool accept = false;
            size_t rand_pos = 0;
            real_t rand_val = 0;
            while (!accept)
            {
                rand_val = curand_uniform(&state) * max_val;
                rand_pos = static_cast<uint32_t>(curand_uniform(&state) * deg);
                if (adj_head + rand_pos >= adj_tail)
                {
                    continue;
                }
                if (rand_val <= min_val)
                {
                    accept = true;
                    break;
                }
                if (g_cache->cache_blocks[cur_index].csr[adj_head + (uint32_t)rand_pos] == prev_vertex)
                {
                    if (rand_val < 1.0 / p)
                        accept = true;
                }
                else if (search(&(g_cache->cache_blocks[prev_index].csr[prev_adj_head]), &(g_cache->cache_blocks[prev_index].csr[prev_adj_tail]), g_cache->cache_blocks[cur_index].csr[adj_head + rand_pos], prev_adj_tail - prev_adj_head + 1))
                {
                    if (rand_val < 1.0)
                        accept = true;
                }
                else
                {
                    if (rand_val < 1.0 / q)
                        accept = true;
                }
            }
            // assert(rand_pos<deg);
            next_vertex = g_cache->cache_blocks[cur_index].csr[adj_head + rand_pos];
        }
        prev_vertex = cur_vertex;
        cur_vertex = next_vertex;
        prev_blk = cur_blk;
        hop++;
        run_step++;
        prev_index = cur_index;
        if (!(cur_vertex >= start_ver && cur_vertex < (start_ver + g_graph->blocks[cur_blk].nverts)))
        {
            cur_index = getindex(g_graph, g_cache, cur_vertex);
            if (cur_index != ncblock)
            {
                cur_blk = g_cache->cache_blocks[cur_index].blk;
            }
            else
                cur_blk = getblk(g_graph, cur_vertex);
        }
#ifdef TEST
        if (cur_blk == prev_blk)
            atomicAdd(&(test->gpu_in_steps), 1);
        else
            atomicAdd(&(test->gpu_pass_steps), 1);
#endif
    }
    if (hop < maxhop)
    {
        g_walks->walk_offset[index] = atomicAdd(&g_walks->block_offset[prev_blk * ncblock + cur_blk], 1);
    }
    else
    {
        g_walks->walk_offset[index] = -1;
    }
    g_walks->walks[index].cur_index = cur_blk;
    g_walks->walks[index].current = cur_vertex;
    g_walks->walks[index].hop = hop;
    g_walks->walks[index].prev_index = prev_blk;
    g_walks->walks[index].previous = prev_vertex;
    return;
}

__global__ void gpu_run(gpu_walks *g_walks, gpu_cache *g_cache, gpu_graph *g_graph, real_t p, real_t q, curandState *states, gpu_test *test)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    bid_t nblock = g_graph->nblock;
    hid_t hops = g_walks->hops;
    wid_t nwalks = g_walks->nwalk;

    for (int i = tid; i < nwalks; i += gridDim.x * blockDim.x)
    {
        gpu_update(i, g_walks, g_cache, g_graph, p, q, hops, nblock, states[tid % (gridDim.x * blockDim.x)], test);
    }
    return;
}

__device__ void SOPR_gpu_update(wid_t index, gpu_walks *g_walks, gpu_cache *g_cache, gpu_graph *g_graph, real_t alpha, hid_t maxhop, bid_t ncblock, curandState state, gpu_test *test)
{
    vid_t cur_vertex = g_walks->walks[index].current;
    vid_t prev_vertex = g_walks->walks[index].previous;
    bid_t cur_blk = g_walks->walks[index].cur_index;
    bid_t prev_blk = g_walks->walks[index].prev_index;
    hid_t hop = g_walks->walks[index].hop;
    bid_t cur_index = g_graph->blocks[cur_blk].index;
    bid_t prev_index = g_graph->blocks[prev_blk].index;
    if (cur_index == ncblock || prev_index == ncblock)
        return;
    wid_t run_step = 0;
    while (cur_index != ncblock && hop < maxhop)
    {
        vid_t start_ver = g_graph->blocks[cur_blk].start_vert;
        vid_t prev_start_ver = g_graph->blocks[prev_blk].start_vert;
        vid_t off = cur_vertex - start_ver;
        vid_t prev_off = prev_vertex - prev_start_ver;
        eid_t adj_head = g_cache->cache_blocks[cur_index].beg_pos[off] - g_graph->blocks[cur_blk].start_edge;
        eid_t adj_tail = g_cache->cache_blocks[cur_index].beg_pos[off + 1] - g_graph->blocks[cur_blk].start_edge;
        eid_t prev_adj_head = g_cache->cache_blocks[prev_index].beg_pos[prev_off] - g_graph->blocks[prev_blk].start_edge;
        eid_t prev_adj_tail = g_cache->cache_blocks[prev_index].beg_pos[prev_off + 1] - g_graph->blocks[prev_blk].start_edge;
        vid_t next_vertex = 0;
        eid_t deg = adj_tail - adj_head;
        eid_t prev_deg = prev_adj_tail - prev_adj_head;
        if (deg == 0)
        {
            hop = maxhop - 1;
        }
        else
        {
            bool accept = false;
            size_t rand_pos = 0;
            real_t rand_val = 0.0;
            real_t max_val = g_max((1.0 - alpha) / deg, (1.0 - alpha) / deg + alpha / prev_deg);
            real_t min_val = g_min((1.0 - alpha) / deg, (1.0 - alpha) / deg + alpha / prev_deg);
            while (!accept)
            {
                rand_val = curand_uniform(&state) * max_val;
                rand_pos = static_cast<uint32_t>(curand_uniform(&state) * deg);
                if (rand_val <= min_val)
                {
                    accept = true;
                    break;
                }
                if (g_cache->cache_blocks[cur_index].csr[adj_head + rand_pos] == prev_vertex)
                {
                    if (rand_val < (1.0 - alpha) / deg)
                        accept = true;
                }
                else if (search(&(g_cache->cache_blocks[prev_index].csr[prev_adj_head]), &(g_cache->cache_blocks[prev_index].csr[prev_adj_tail]), g_cache->cache_blocks[cur_index].csr[adj_head + rand_pos], prev_adj_tail - prev_adj_head + 1))
                {
                    if (rand_val < ((1.0 - alpha) / deg + alpha / prev_deg))
                        accept = true;
                }
                else
                {
                    if (rand_val < (1.0 - alpha) / deg)
                        accept = true;
                }
            }
            next_vertex = g_cache->cache_blocks[cur_index].csr[adj_head + rand_pos];
        }
        prev_vertex = cur_vertex;
        cur_vertex = next_vertex;
        prev_blk = cur_blk;
        hop++;
        run_step++;
        prev_index = cur_index;
        if (!(cur_vertex >= start_ver && cur_vertex < (start_ver + g_graph->blocks[cur_blk].nverts)))
        {
            cur_index = getindex(g_graph, g_cache, cur_vertex);
            if (cur_index != ncblock)
            {
                cur_blk = g_cache->cache_blocks[cur_index].blk;
            }
            else
                cur_blk = getblk(g_graph, cur_vertex);
        }
#ifdef TEST
        if (cur_blk == prev_blk)
            atomicAdd(&(test->gpu_in_steps), 1);
        else
            atomicAdd(&(test->gpu_pass_steps), 1);
#endif
    }
    if (hop < maxhop)
    {
        g_walks->walk_offset[index] = atomicAdd(&g_walks->block_offset[prev_blk * ncblock + cur_blk], 1);
    }
    else
    {
        g_walks->walk_offset[index] = -1;
    }
    g_walks->walks[index].cur_index = cur_blk;
    g_walks->walks[index].current = cur_vertex;
    g_walks->walks[index].hop = hop;
    g_walks->walks[index].prev_index = prev_blk;
    g_walks->walks[index].previous = prev_vertex;
    return;
}

__global__ void SOPR_gpu_run(gpu_walks *g_walks, gpu_cache *g_cache, gpu_graph *g_graph, real_t alpha, curandState *states, gpu_test *test)
{
    bid_t nblock = g_graph->nblock;
    hid_t hops = g_walks->hops;
    wid_t nwalks = g_walks->nwalk;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nwalks; i += gridDim.x * blockDim.x)
    {
        SOPR_gpu_update(i, g_walks, g_cache, g_graph, alpha, hops, nblock, states[i % (gridDim.x * blockDim.x)], test);
    }
    return;
}

__global__ void prefix(gpu_walks *g_walks, gpu_graph *g_graph)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    bid_t nblocks = g_graph->nblock;
    if (tid == 0)
    {
        size_t p = 0;
        size_t num = 0;
        for (int i = 0; i < nblocks * nblocks; i++)
        {
            num = g_walks->block_offset[i];
            g_walks->block_offset[i] = p;
            p += num;
        }
        g_walks->block_offset[nblocks * nblocks] = p;
        g_walks->res_nwalk = p;
    }
}

__global__ void insertglobal(gpu_walks *g_walks, gpu_cache *g_cache, gpu_graph *g_graph)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    wid_t nwalks = g_walks->nwalk;
    bid_t nblocks = g_graph->nblock;
    bid_t blockindex;
    walker_t walks;
    uint32_t index;
    for (int i = tid; i < nwalks; i += gridDim.x * blockDim.x)
    {
        walks = g_walks->walks[i];
        index = g_walks->walk_offset[i];
        blockindex = walks.prev_index * nblocks + walks.cur_index;
        if (index != -1)
        {
            wid_t offset = g_walks->block_offset[blockindex] + index;
            g_walks->res_walks[offset] = walks;
        }
    }
    return;
}

__global__ void walkswap(gpu_walks *g_walks)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
    {
        walker_t *p = g_walks->walks;
        g_walks->walks = g_walks->res_walks;
        g_walks->res_walks = p;
        g_walks->res_nwalk = 0;
    }
}

// zerocopy gpu update
__global__ void zerocopy_update(gpu_walks *g_walks, cache_block *d_map, gpu_graph *g_graph, real_t p, real_t q, bid_t nblocks, curandState *states, gpu_test *test)
{
    curandState state = states[blockIdx.x * blockDim.x + threadIdx.x];
    hid_t maxhop = g_walks->hops;
    uint32_t run_step = 0;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < g_walks->nwalk; tid += blockDim.x * gridDim.x)
    {
        walker_t walk = g_walks->walks[tid];
        vid_t cur_vertex = walk.current;
        vid_t prev_vertex = walk.previous;
        bid_t cur_blk = walk.cur_index;
        bid_t prev_blk = walk.prev_index;
        hid_t hop = walk.hop;
        while (hop < maxhop)
        {
            vid_t start_ver = g_graph->blocks[cur_blk].start_vert;
            vid_t prev_start_ver = g_graph->blocks[prev_blk].start_vert;
            vid_t off = cur_vertex - start_ver;
            vid_t prev_off = prev_vertex - prev_start_ver;
            eid_t adj_head = d_map[cur_blk].beg_pos[off] - g_graph->blocks[cur_blk].start_edge;
            eid_t adj_tail = d_map[cur_blk].beg_pos[off + 1] - g_graph->blocks[cur_blk].start_edge;
            eid_t prev_adj_head = d_map[prev_blk].beg_pos[prev_off] - g_graph->blocks[prev_blk].start_edge;
            eid_t prev_adj_tail = d_map[prev_blk].beg_pos[prev_off + 1] - g_graph->blocks[prev_blk].start_edge;
            vid_t next_vertex = 0;
            eid_t deg = adj_tail - adj_head;
            if (deg == 0)
            {
                hop = maxhop - 1;
            }
            else
            {
                // if(deg>=g_graph->blocks[cur_blk].nedges) return;
                real_t max_val = g_max(1.0 / p, g_max(1.0 / q, 1.0));
                real_t min_val = g_min(1.0 / p, g_min(1.0 / q, 1.0));
                bool accept = false;
                size_t rand_pos = 0;
                real_t rand_val = 0;
                while (!accept)
                {
                    rand_val = curand_uniform(&state) * max_val;
                    rand_pos = static_cast<uint32_t>(curand_uniform(&state) * deg);
                    if (rand_val <= min_val)
                    {
                        accept = true;
                        break;
                    }
                    if (d_map[cur_blk].csr[adj_head + rand_pos] == prev_vertex)
                    {
                        if (rand_val < 1.0 / p)
                            accept = true;
                    }
                    else if (search(&(d_map[prev_blk].csr[prev_adj_head]), &(d_map[prev_blk].csr[prev_adj_tail]), d_map[cur_blk].csr[adj_head + rand_pos], prev_adj_tail - prev_adj_head + 1))
                    {
                        if (rand_val < 1.0)
                            accept = true;
                    }
                    else
                    {
                        if (rand_val < 1.0 / q)
                            accept = true;
                    }
                }
                // assert(rand_pos<deg);
                next_vertex = d_map[cur_blk].csr[adj_head + rand_pos];
            }
            prev_vertex = cur_vertex;
            cur_vertex = next_vertex;
            prev_blk = cur_blk;
            hop++;
            run_step++;
            if (!(cur_vertex >= start_ver && cur_vertex < (start_ver + g_graph->blocks[cur_blk].nverts)))
            {
                cur_blk = zerocopy_getblk(nblocks, cur_vertex, g_graph);
            }
#ifdef TEST
            if (cur_blk != prev_blk)
            {
                atomicAdd(&(test->gpu_in_steps), 1);
            }
            else
            {
                atomicAdd(&(test->gpu_pass_steps), 1);
            }
#endif
        }
    }
    return;
}

// zerocopy gpu update
__global__ void SOPR_zerocopy_update(gpu_walks *g_walks, cache_block *d_map, gpu_graph *g_graph, real_t alpha, bid_t nblocks, curandState *states, gpu_test *test)
{
    curandState state = states[blockIdx.x * blockDim.x + threadIdx.x];
    hid_t maxhop = g_walks->hops;
    uint32_t run_step = 0;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < g_walks->nwalk; tid += blockDim.x * gridDim.x)
    {
        walker_t walk = g_walks->walks[tid];
        vid_t cur_vertex = walk.current;
        vid_t prev_vertex = walk.previous;
        bid_t cur_blk = walk.cur_index;
        bid_t prev_blk = walk.prev_index;
        hid_t hop = walk.hop;
        while (hop < maxhop)
        {
            vid_t start_ver = g_graph->blocks[cur_blk].start_vert;
            vid_t prev_start_ver = g_graph->blocks[prev_blk].start_vert;
            vid_t off = cur_vertex - start_ver;
            vid_t prev_off = prev_vertex - prev_start_ver;
            eid_t adj_head = d_map[cur_blk].beg_pos[off] - g_graph->blocks[cur_blk].start_edge;
            eid_t adj_tail = d_map[cur_blk].beg_pos[off + 1] - g_graph->blocks[cur_blk].start_edge;
            eid_t prev_adj_head = d_map[prev_blk].beg_pos[prev_off] - g_graph->blocks[prev_blk].start_edge;
            eid_t prev_adj_tail = d_map[prev_blk].beg_pos[prev_off + 1] - g_graph->blocks[prev_blk].start_edge;
            vid_t next_vertex = 0;
            eid_t deg = adj_tail - adj_head;
            eid_t prev_deg = prev_adj_tail - prev_adj_head;
            eid_t max_deg = g_max(deg, prev_deg);
            if (deg == 0)
            {
                hop = maxhop - 1;
            }
            else
            {
                bool accept = false;
                size_t rand_pos = 0;
                real_t rand_val = 0.0;
                real_t max_val = g_max((1.0 - alpha) / deg, (1.0 - alpha) / deg + alpha / prev_deg);
                real_t min_val = g_min((1.0 - alpha) / deg, (1.0 - alpha) / deg + alpha / prev_deg);
                while (!accept)
                {
                    rand_val = curand_uniform(&state) * max_val;
                    rand_pos = static_cast<uint32_t>(curand_uniform(&state) * deg);
                    if (rand_val <= min_val)
                    {
                        accept = true;
                        break;
                    }
                    if (d_map[cur_blk].csr[adj_head + rand_pos] == prev_vertex)
                    {
                        if (rand_val < (1.0 - alpha) / deg)
                            accept = true;
                    }
                    else if (search(&(d_map[prev_blk].csr[prev_adj_head]), &(d_map[prev_blk].csr[prev_adj_tail]), d_map[cur_blk].csr[adj_head + rand_pos], prev_adj_tail - prev_adj_head + 1))
                    {
                        if (rand_val < ((1.0 - alpha) / deg + alpha / prev_deg))
                            accept = true;
                    }
                    else
                    {
                        if (rand_val < (1.0 - alpha) / deg)
                            accept = true;
                    }
                }
                next_vertex = d_map[cur_blk].csr[adj_head + rand_pos];
            }
            prev_vertex = cur_vertex;
            cur_vertex = next_vertex;
            prev_blk = cur_blk;
            hop++;
            run_step++;
            if (!(cur_vertex >= start_ver && cur_vertex < (start_ver + g_graph->blocks[cur_blk].nverts)))
            {
                cur_blk = zerocopy_getblk(nblocks, cur_vertex, g_graph);
            }
#ifdef TEST
            if (cur_blk != prev_blk)
            {
                atomicAdd(&(test->gpu_in_steps), 1);
            }
            else
            {
                atomicAdd(&(test->gpu_pass_steps), 1);
            }
#endif
        }
    }
    return;
}
