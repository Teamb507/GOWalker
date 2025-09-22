#ifndef _GRAPH_CACHE_H_
#define _GRAPH_CACHE_H_
#include <cstdlib>
#include <cassert>
#include <mutex>
#include <memory>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "api/constants.hpp"
#include "api/types.hpp"
#include "util/util.hpp"
#include "util/hash.hpp"
#include "util/io.hpp"
#include "config.hpp"

/**
 * This file contribute to define graph block cache structure and some operations
 * on the cache, such as swapin and swapout, schedule blocks.
 */

 /** block has four state
  *
  * `USING`      : the block is running
  * `USED`       : the block is finished runing, but still in memory
  * `ACTIVE`     : the block is in memroy, but not use
  * `INACTIVE`   : the block is in disk
  */



class block_t
{
public:
    bid_t blk, cache_index;   /* the block number, and the memory index */
    vid_t start_vert, nverts; /* block start vertex and the number of vertex in block */
    eid_t start_edge, nedges; /* block start edge and the number of edges in this block */

    block_state status;
    block_t()
    {
        blk = cache_index = 0;
        start_vert = nverts = 0;
        start_edge = nedges = 0;
        status = INACTIVE;
    }

    block_t& operator=(const block_t& other)
    {
        if (this != &other)
        {
            this->blk = other.blk;
            this->start_vert = other.start_vert;
            this->nverts = other.nverts;
            this->start_edge = other.start_edge;
            this->nedges = other.nedges;
            this->status = other.status;
        }
        return *this;
    }
};

class cache_block
{
public:
    block_t* block;
    bool cudazerocopy;
    eid_t* beg_pos;
    vid_t* degree;
    vid_t* csr;
    real_t* weights;

    cache_block()
    {
        block = NULL;
        beg_pos = NULL;
        degree = NULL;
        csr = NULL;
        weights = NULL;
        cudazerocopy = false;
    }

    ~cache_block()
    {
        if (beg_pos)
            checkCudaError(cudaFreeHost(beg_pos));
        if (degree)
            checkCudaError(cudaFreeHost(degree));
        if (csr)
            checkCudaError(cudaFreeHost(csr));
        if (weights)
            checkCudaError(cudaFreeHost(weights));
    }
    void cudaregister()
    {
        if (!cudazerocopy)
        {
            cudaHostRegister(block, sizeof(block_t), cudaHostRegisterMapped);
            cudaHostRegister(beg_pos, sizeof(eid_t) * (block->nverts + 1), cudaHostRegisterMapped);
            cudaHostRegister(csr, sizeof(vid_t) * block->nedges, cudaHostRegisterMapped);
            if (weights != NULL)
                cudaHostRegister(weights, sizeof(real_t) * block->nedges, cudaHostRegisterMapped);
            cudazerocopy = true;
        }
    }
    void cudamalloc(bool weighted, size_t blocksize)
    {
        if (!cudazerocopy)
        {
            if (beg_pos != NULL)
                free(beg_pos);
            if (csr != NULL)
                free(csr);
            checkCudaError(cudaHostAlloc((void**)&beg_pos, blocksize, cudaHostAllocMapped));
            checkCudaError(cudaHostAlloc((void**)&csr, blocksize, cudaHostAllocMapped));
            if (weighted)
                checkCudaError(cudaHostAlloc((void**)&weights, blocksize, cudaHostAllocMapped));
            cudazerocopy = true;
        }
    }
};

void swap(cache_block& cb1, cache_block& cb2)
{
    block_t* tblock = cb2.block;
    eid_t* tbeg_pos = cb2.beg_pos;
    vid_t* tdegree = cb2.degree;
    vid_t* tcsr = cb2.csr;
    real_t* tw = cb2.weights;

    cb2.block = cb1.block;
    cb2.beg_pos = cb1.beg_pos;
    cb2.degree = cb1.degree;
    cb2.csr = cb1.csr;
    cb2.weights = cb1.weights;

    cb1.block = tblock;
    cb1.beg_pos = tbeg_pos;
    cb1.degree = tdegree;
    cb1.csr = tcsr;
    cb1.weights = tw;
}

class graph_block
{
public:
    bid_t nblocks;
    block_t* blocks;

    graph_block(graph_config* conf)
    {
        std::string vert_block_name = get_vert_blocks_name(conf->base_name, conf->blocksize);
        std::string edge_block_name = get_edge_blocks_name(conf->base_name, conf->blocksize);

        std::vector<vid_t> vblocks = load_graph_blocks<vid_t>(vert_block_name);
        std::vector<eid_t> eblocks = load_graph_blocks<eid_t>(edge_block_name);
        eid_t maxedges = (eid_t)conf->blocksize / sizeof(vid_t);
        vid_t maxverts = (vid_t)conf->blocksize / sizeof(eid_t);
        nblocks = vblocks.size() - 1;
        checkCudaError(cudaHostAlloc((void**)&blocks, sizeof(block_t) * nblocks, cudaHostAllocMapped));

        for (bid_t blk = 0; blk < nblocks; blk++)
        {
            blocks[blk].blk = blk;
            blocks[blk].cache_index = nblocks;
            blocks[blk].start_vert = vblocks[blk];
            blocks[blk].nverts = vblocks[blk + 1] - vblocks[blk];
            blocks[blk].start_edge = eblocks[blk];
            blocks[blk].nedges = eblocks[blk + 1] - eblocks[blk];
            if (blocks[blk].nverts == 1 && blocks[blk].nedges > maxedges) {
                blocks[blk].nedges = maxedges;
            }
            blocks[blk].status = INACTIVE;
        }
    }

    block_t& operator[](bid_t blk)
    {
        assert(blk < nblocks);
        return blocks[blk];
    }

    bid_t get_block(vid_t v)
    {
        bid_t blk = 0;
        for (; blk < nblocks; blk++)
        {
            if (v < blocks[blk].start_vert + blocks[blk].nverts)
                return blk;
        }
        return nblocks;
    }
};

class graph_cache
{
public:
    bid_t ncblock;                         /* number of cache blocks */
    std::vector<cache_block> cache_blocks; /* the cached blocks */
    std::vector<bid_t> walk_blocks;
    graph_cache()
    {
        ncblock = 0;
    }

    graph_cache(bid_t nblocks, size_t blocksize)
    {
        ncblock = nblocks;
        assert(ncblock > 0);
        cache_blocks.resize(ncblock);
    }
    cache_block& operator[](size_t index)
    {
        assert(index < ncblock);
        return cache_blocks[index];
    }

    cache_block operator[](size_t index) const
    {
        assert(index < ncblock);
        return cache_blocks[index];
    }
};

#endif
