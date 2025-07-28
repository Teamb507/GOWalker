#ifndef _GRAPH_WALK_H_
#define _GRAPH_WALK_H_
#include <algorithm>
#include "api/types.hpp"
#include "api/graph_buffer.hpp"
#include "util/hash.hpp"
#include "cache.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thread>
#include <atomic>
#include <mutex>
class block_desc_manager_t
{
private:
    int desc;

public:
    block_desc_manager_t(const std::string&& file_name)
    {
        desc = open(file_name.c_str(), O_RDWR | O_CREAT | O_APPEND, S_IROTH | S_IWOTH | S_IWUSR | S_IRUSR);
    }
    ~block_desc_manager_t()
    {
        if (desc > 0)
            close(desc);
    }
    int get_desc() const { return desc; }
};

__global__ void generatewalks(size_t walkpersource, vid_t minnode, vid_t numnode, walker_t* walks, bid_t blk)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numnode)
    {
        for (int i = 0; i < walkpersource; i++)
        {
            walks[tid * walkpersource + i].source = minnode + tid;
            walks[tid * walkpersource + i].current = minnode + tid;
            walks[tid * walkpersource + i].previous = minnode + tid;
            walks[tid * walkpersource + i].cur_index = blk;
            walks[tid * walkpersource + i].prev_index = blk;
            walks[tid * walkpersource + i].hop = 0;
            walks[tid * walkpersource + i].id = (minnode + tid) * walkpersource + i;
        }
    }
    return;
}

__global__ void SOPR_generatewalks(vid_t source, wid_t numwalks, wid_t idoffset, walker_t* walks, bid_t blk)
{
    for (wid_t i = 0;i < numwalks;i += gridDim.x * blockDim.x)
    {
        walks[i].source = source;
        walks[i].current = source;
        walks[i].previous = source;
        walks[i].cur_index = blk;
        walks[i].prev_index = blk;
        walks[i].hop = 0;
        walks[i].id = idoffset + i;
    }
    return;
}

__global__ void lt_pushwalkoffset(walker_t* walks, int oldoffset, int offset, int nwalks)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) % nwalks;
    walks[offset + tid] = walks[oldoffset + tid];
    return;
}

class walk_block
{
public:
    wid_t num;
    graph_buffer<walker_t>* buffer;
    std::mutex mtx;

    wid_t cpu_batch;
    ~walk_block()
    {
        num = 0;
        if (buffer != NULL)
        {
            free(buffer);
        }
    }

    void set(wid_t batchsize)
    {
        num = 0;
        cpu_batch = batchsize;
        buffer = NULL;
    }

    wid_t load(bool batch, graph_buffer<walker_t>** walks)
    {
        wid_t load = 0;
        mtx.lock();
        if (batch)
        {
            if (num > 0)
            {
                auto p = buffer;
                load += p->bsize;
                num -= p->bsize;
                buffer = buffer->next;
                p->next = *walks;
                *walks = p;
            }
            mtx.unlock();
            return load;
        }
        else
        {
            if (num > 0)
            {
                if ((*walks) != NULL)
                    return 0;
                *walks = buffer;
                buffer = NULL;
                load += num;
                num = 0;
                mtx.unlock();
                return load;
            }
        }
        mtx.unlock();
        return 0;
    }
    wid_t insert(bool device, walker_t* walks, wid_t numwalks, cudaStream_t stream)
    {
        wid_t load = 0;
        mtx.lock();
        if (!device)
        {
            graph_buffer<walker_t>* c = (graph_buffer<walker_t> *)malloc(sizeof(graph_buffer<walker_t>));
            c->array = walks;
            c->bsize = numwalks;
            c->capacity = numwalks;
            c->next = buffer;
            buffer = c;
            num += numwalks;
            load += numwalks;
            mtx.unlock();
            return load;
        }
        else
        {
            while (load < numwalks)
            {
                walker_t* array = NULL;
                checkCudaError(cudaHostAlloc((void**)&array, sizeof(walker_t) * cpu_batch, cudaHostAllocMapped));
                wid_t batch = cpu_batch > (numwalks - load) ? (numwalks - load) : cpu_batch;
                checkCudaError(cudaMemcpyAsync(array, &walks[load], sizeof(walker_t) * batch, cudaMemcpyDeviceToHost, stream));
                graph_buffer<walker_t>* c = (graph_buffer<walker_t> *)malloc(sizeof(graph_buffer<walker_t>));
                c->array = array;
                c->bsize = batch;
                c->capacity = batch;
                c->next = buffer;
                buffer = c;
                num += batch;
                load += batch;
            }
        }
        mtx.unlock();
        return load;
    }

    wid_t block_numwalks()
    {
        mtx.lock();
        wid_t numwalks = num;
        mtx.unlock();
        return numwalks;
    }
};

class graph_walk
{
public:
    size_t numwalks;
    wid_t cpu_batch;
    wid_t gpu_batch;
    vid_t nvertices;
    vid_t minvert;
    bid_t nblocks, totblocks;
    tid_t nthreads;
    std::mutex forbid_mtx;
    std::vector<bid_t> forbid;
    graph_buffer<walker_t>* g_walks; 
    graph_buffer<walker_t>* c_walks; 
    walk_block* block_walks;
    graph_buffer<walker_t>** cpu_walkbuffer;
    graph_block* global_blocks;

    graph_walk(graph_config& conf, graph_driver& driver, graph_block& blocks)
    {
        cpu_batch = conf.cpu_batch;
        gpu_batch = conf.gpu_batch;
        numwalks = conf.numwalks;
        nvertices = conf.nvertices;
        minvert = conf.min_vert;
        nthreads = conf.cpu_threads;
        global_blocks = &blocks;

        nblocks = global_blocks->nblocks;
        totblocks = nblocks * nblocks;
        g_walks = NULL;
        c_walks = NULL;
        block_walks = new walk_block[totblocks];
        for (int i = 0; i < totblocks; i++)
        {
            block_walks[i].set(cpu_batch);
        }
        cpu_walkbuffer = (graph_buffer<walker_t> **)malloc(sizeof(graph_buffer<walker_t> *) * totblocks);
        for (int i = 0; i < totblocks; i++)
        {
            cpu_walkbuffer[i] = (graph_buffer<walker_t> *)malloc(sizeof(graph_buffer<walker_t>) * nthreads);
            for (int j = 0; j < nthreads; j++)
            {
                cpu_walkbuffer[i][j].alloc(cpu_batch);
            }
        }
    }

    ~graph_walk()
    {
        delete[] block_walks;
        free(g_walks);
        free(c_walks);
    }
    wid_t gpuinsertbatchwalk(walker_t* gpuwalks, size_t numwalks, bid_t blk, cudaStream_t stream)
    {
        size_t load = 0;
        load += block_walks[blk].insert(1, gpuwalks, numwalks, stream);
        return load;
    }
    wid_t gpu_createwalk(size_t walkpersource, size_t numwalks, graph_config* conf, cudaStream_t stream)
    {
        walker_t* gpuwalks;
        if (conf->algorithm == SOPR)
        {
            std::cout<<"SOPR create walks!"<<std::endl;
            if (walkpersource > 0)
            {
                checkCudaError(cudaMallocAsync((void**)&gpuwalks, sizeof(walker_t) * cpu_batch * walkpersource, stream));
                size_t numwalks = conf->numwalks;
                size_t maxbatch = cpu_batch;
                size_t num = 0;
                while (num + maxbatch < numwalks)
                {
                    SOPR_generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (1, maxbatch, num, gpuwalks, 0);
                    num += maxbatch;
                    gpuinsertbatchwalk(gpuwalks, maxbatch, 0, stream);
                }
                if (num < numwalks)
                {
                    SOPR_generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (1, numwalks - num, num, gpuwalks, 0);
                    gpuinsertbatchwalk(gpuwalks, numwalks - num, 0, stream);
                }

            }
        }
        else {
            if (walkpersource > 0)
            {
                checkCudaError(cudaMallocAsync((void**)&gpuwalks, sizeof(walker_t) * cpu_batch * walkpersource, stream));
                size_t maxbatch = cpu_batch;
                vid_t minnode = 0;
                vid_t maxnode = 0;
                for (bid_t blk = 0; blk < nblocks; blk++)
                {
                    size_t num = 0;
                    minnode = global_blocks->blocks[blk].start_vert;
                    while (num + maxbatch < global_blocks->blocks[blk].nverts)
                    {
                        generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (walkpersource, minnode, maxbatch, gpuwalks, blk);
                        num += maxbatch;
                        minnode += maxbatch;
                        gpuinsertbatchwalk(gpuwalks, maxbatch * walkpersource, blk * nblocks + blk, stream);
                    }
                    if (num < global_blocks->blocks[blk].nverts)
                    {
                        generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (walkpersource, minnode, (global_blocks->blocks[blk].nverts - num), gpuwalks, blk);
                        gpuinsertbatchwalk(gpuwalks, (global_blocks->blocks[blk].nverts - num) * walkpersource, blk * nblocks + blk, stream);
                    }
                }
            }
            else
            {
                size_t batch = (numwalks / nblocks + 1) > cpu_batch ? cpu_batch : (numwalks / nblocks + 1);
                checkCudaError(cudaMallocAsync((void**)&gpuwalks, sizeof(walker_t) * batch * 2, stream));
                vid_t minnode = 0;
                vid_t maxnode = 0;
                wid_t generated = 0;
                while (generated < numwalks)
                {
                    for (bid_t blk = 0; blk < nblocks; blk++)
                    {
                        size_t num = 0;
                        srand(time(0));

                        minnode = global_blocks->blocks[blk].start_vert + rand() % 1000;
                        if (generated + batch < numwalks)
                        {
                            generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (1, minnode, batch, gpuwalks, blk);
                            gpuinsertbatchwalk(gpuwalks, batch, blk * nblocks + blk, stream);
                            generated += batch;
                        }
                        else if (generated < numwalks)
                        {
                            generatewalks << <conf->blockpergrid, conf->threadperblock, 0, stream >> > (1, minnode, numwalks - generated, gpuwalks, blk);
                            gpuinsertbatchwalk(gpuwalks, numwalks - generated, blk * nblocks + blk, stream);
                            generated += (numwalks - generated);
                        }
                    }
                }
            }
        }
        cudaStreamSynchronize(stream);
        cudaFreeAsync(gpuwalks, stream);
        return 0;
    }

    wid_t cpuinsertbatchwalk(walker_t* cpuwalks, size_t numwalks, bid_t blk)
    {
        return block_walks[blk].insert(0, cpuwalks, numwalks, 0);
    }
    wid_t nblockwalks(bid_t blk)
    {
        wid_t walksum = 0;
        walksum = block_walks[blk].block_numwalks();
        return walksum;
    }
    wid_t nwalks() 
    {
        wid_t walksum = 0;
        for (bid_t blk = 0; blk < totblocks; blk++)
        {
            wid_t num = this->nblockwalks(blk);
            walksum += num;
        }
        return walksum;
    }
    bool test_finished_walks()
    {
        return this->nwalks() == 0;
    }

    wid_t zerocopywalktogpu(bid_t blk)
    {
        if (g_walks != NULL)
        {
            std::cout << "last epoch update error!" << std::endl;
        }
        wid_t load = 0;

        if (block_walks[blk].block_numwalks() > 0)
        {
            load += block_walks[blk].load(0, &g_walks);
        }
        return load;
    }

    wid_t nbufferwalks()
    {
        wid_t walknum = 0;
        for (int i = 0; i < totblocks; i++)
        {
            walknum += block_walks[i].block_numwalks();
        }
        return walknum;
    }

    bool test_buffer_over()
    {
        for (bid_t blk = 0; blk < totblocks; blk++)
        {
            if (this->block_walks[blk].block_numwalks() > 0)
                return false;
        }
        return true;
    }

    bid_t cpu_loadblkwalk(bid_t blk)
    {
        wid_t load = block_walks[blk].load(1, &c_walks);
        return load;
    }

    wid_t gpu_loadwalk_pipeline(bid_t blk, wid_t maxwalks, wid_t offset, walker_t* gpu_buffer, cudaStream_t stream)
    {
        wid_t load = 0;
        while (block_walks[blk].block_numwalks() > 0)
        {
            wid_t batch = block_walks[blk].load(1, &g_walks);
            if (load + batch > maxwalks)
            {
                block_walks[blk].insert(0, g_walks->array, batch, stream);
                free(g_walks);
                g_walks = NULL;
                return load;
            }
            checkCudaError(cudaMemcpyAsync(&gpu_buffer[offset + load], g_walks->array, sizeof(walker_t) * batch, cudaMemcpyHostToDevice, stream));
            checkCudaError(cudaStreamSynchronize(stream));
            load += batch;
            checkCudaError(cudaFreeHost(g_walks->array));
            free(g_walks);
            g_walks = NULL;
        }
        return load;
    }

    wid_t copy_back(gpu_walks* gpu_walkbuffer, graph_cache* cache, cudaStream_t stream, metrics& _m)
    {
        checkCudaError(cudaStreamSynchronize(stream));
        gpu_walks* cpu_walks = (gpu_walks*)malloc(sizeof(gpu_walks));
        checkCudaError(cudaMemcpyAsync(cpu_walks, gpu_walkbuffer, sizeof(gpu_walks), cudaMemcpyDeviceToHost, stream));
        wid_t* offset = (wid_t*)malloc(sizeof(wid_t) * (totblocks + 1));
        checkCudaError(cudaMemcpyAsync(offset, cpu_walks->block_offset, sizeof(wid_t) * (totblocks + 1), cudaMemcpyDeviceToHost, stream));
        wid_t* walknum = (wid_t*)malloc(sizeof(wid_t) * totblocks);
        wid_t p = 0;
        std::vector<bid_t> cached_walk_block;
        cached_walk_block = cache->walk_blocks;
        for (int i = 0; i < nblocks * nblocks; i++)
        {
            walknum[i] = offset[i + 1] - offset[i];
        }
        int pos = 0;
        for (int i = 0; i < nblocks; i++)
        {
            for (int j = 0; j < nblocks; j++)
            {
                bid_t blk = i * nblocks + j;
                if (pos < cached_walk_block.size())
                {
                    if (blk == cached_walk_block[pos])
                    {
                        pos++;
                        if (walknum[blk] > 0)
                        {
                            checkCudaError(cudaMemcpyAsync(&cpu_walks->walks[p], &cpu_walks->walks[offset[blk]], sizeof(walker_t) * walknum[blk], cudaMemcpyDeviceToDevice, stream));
                            p += walknum[blk];
                            continue;
                        }
                    }
                }
                if (walknum[blk] > 0)
                {
                    gpuinsertbatchwalk(&cpu_walks->walks[offset[blk]], walknum[blk], blk, stream);
                }
            }
        }
        return p;
    }

    wid_t copywalker(gpu_walks* gpu_walkbuffer, wid_t nwalk, wid_t offset, graph_cache* cache, graph_config* conf, cudaStream_t stream, metrics& _m)
    {
        bid_t nblocks = nblocks;
        static gpu_walks* cpu_walks = (gpu_walks*)malloc(sizeof(gpu_walks));
        checkCudaError(cudaMemcpyAsync(cpu_walks, gpu_walkbuffer, sizeof(gpu_walks), cudaMemcpyDeviceToHost, stream));
        std::vector<bid_t> cached_walk_block;
        cached_walk_block = cache->walk_blocks;
        for (int i = 0; i < cached_walk_block.size(); i++)
        {
            if (offset < nwalk)
            {
                offset += gpu_loadwalk_pipeline(cached_walk_block[i], nwalk - offset, offset, cpu_walks->walks, stream);
            }
            else {
                break;
            }
        }
        cpu_walks->nwalk = offset;
        cpu_walks->res_nwalk = 0;
        checkCudaError(cudaMemcpyAsync(gpu_walkbuffer, cpu_walks, sizeof(gpu_walks), cudaMemcpyHostToDevice, stream));
        checkCudaError(cudaMemsetAsync(cpu_walks->block_offset, 0, sizeof(wid_t) * (totblocks + 1), stream));
        return offset;
    }


    wid_t blockstobuffer()
    {
        for (int i = 0; i < totblocks; i++)
        {
            for (int tid = 0; tid < nthreads; tid++)
            {
                block_walks[i].insert(0, cpu_walkbuffer[i][tid].array, cpu_walkbuffer[i][tid].bsize, 0);
            }
            free(cpu_walkbuffer[i]);
        }
    }
};
#endif
