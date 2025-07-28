#ifndef _GRAPH_TYPES_H_
#define _GRAPH_TYPES_H_
#include <omp.h>
#include <stdint.h>
#include <functional>
#include <vector>
#include <climits>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <atomic>
typedef uint32_t vid_t;  /* vertex id */
typedef uint64_t eid_t;  /* edge id */
typedef uint32_t bid_t;  /* block id */
typedef uint32_t rank_t; /* block rank */
typedef uint32_t hid_t;  /* walk hop */
typedef uint16_t tid_t;  /* thread id */
typedef uint32_t wid_t;  /* walk id */
typedef uint64_t walk_t; /* walker data type */
typedef float real_t;    /* edge weight */
#define HOP_SIZE 8 /* hop field size */
#define zerocopythresh 4096
#define VERTEX_SIZE 28    /* source, previous, current vertex size */
#define WALKER_ID_SIZE 36 /* walker id size */
#define walkperthread 16
#define sharedsize 1024
typedef enum { node2vec, SOPR } AlgorithmType;
void checkCudaError(const cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "cuda false:" << std::endl;
        exit(EXIT_FAILURE);
    }
}
enum WeightType
{
    UNWEIGHTED,
    WEIGHTED
};
enum block_state
{
    USING = 1,
    USED,
    ACTIVE,
    INACTIVE,
    PART
};
struct walker_t
{
    wid_t id;
    vid_t source, previous, current;
    hid_t hop;
    bid_t cur_index, prev_index; // blk
    bid_t totblk;
};

walker_t walker_makeup(wid_t id, vid_t source, vid_t previous, vid_t pos, hid_t hop, bid_t c_index, bid_t p_index)
{
    walker_t walk_data;
    walk_data.id = id;
    walk_data.source = source;
    walk_data.previous = previous;
    walk_data.current = pos;
    walk_data.hop = hop;
    walk_data.cur_index = c_index;
    walk_data.prev_index = p_index;
    return walk_data;
}

struct gpu_test
{
    uint32_t gpu_in_steps;
    uint32_t gpu_pass_steps;
};

class graph_test
{
public:
    uint64_t gpu_in_steps;
    uint64_t gpu_pass_steps;
    uint64_t cpu_allsteps;
    uint64_t* cpu_steps;
    std::vector<double>* thread_time;
    gpu_test* h_gpu;
    gpu_test* d_gpu;
    std::vector<std::pair<eid_t, uint32_t>> IO_uti;
    graph_test()
    {
        gpu_in_steps = 0;
        gpu_pass_steps = 0;
        cpu_allsteps = 0;
        cpu_steps = (uint64_t*)malloc(sizeof(uint64_t) * omp_get_max_threads());
        thread_time = (std::vector<double> *)malloc(sizeof(std::vector<double>) * omp_get_max_threads());
        h_gpu = (gpu_test*)malloc(sizeof(gpu_test));
        IO_uti.clear();
        cudaMalloc((void**)&d_gpu, sizeof(gpu_test));
    }
    ~graph_test()
    {
        cudaFree(d_gpu);
    }
    void copygputocpu()
    {
        cudaMemcpy(&h_gpu->gpu_in_steps, &(d_gpu->gpu_in_steps), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_gpu->gpu_pass_steps, &(d_gpu->gpu_pass_steps), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemset(d_gpu, 0, sizeof(gpu_test));
    }
    void aggregate_gpu()
    {
        copygputocpu();
        gpu_in_steps += h_gpu->gpu_in_steps;
        gpu_pass_steps += h_gpu->gpu_pass_steps;
    }
    void aggregate_cpu(int threadnum)
    {
        for (int i = 0;i < threadnum;i++)
        {
            cpu_allsteps += cpu_steps[i];
        }
    }
    void utilization_rate(eid_t edgenum)
    {
        aggregate_gpu();
        if (edgenum == 0)
        {
            IO_uti[IO_uti.size() - 1].second += (h_gpu->gpu_in_steps + h_gpu->gpu_pass_steps);
            return;
        }
        IO_uti.push_back(std::make_pair(edgenum, h_gpu->gpu_in_steps + h_gpu->gpu_pass_steps));
    }
};
class gpu_block
{
public:
    bid_t blk;
    eid_t* beg_pos;
    vid_t* csr;
    real_t* weights;
};
class gpu_cache
{
public:
    bid_t ncblock;
    gpu_block* cache_blocks;
};
class gpu_walks
{
public:
    hid_t hops;
    wid_t nwalk;
    wid_t res_nwalk;
    wid_t* walk_offset; // xinzeng
    walker_t* walks;
    walker_t* res_walks;
    wid_t* block_offset;
};

class gpu_graph_block
{
public:
    bid_t blk, index;
    vid_t start_vert, nverts;
    eid_t start_edge, nedges;
    block_state status;
};
class gpu_graph
{
public:
    gpu_graph_block* blocks;
    bid_t nblock;
};

class gpu_stream
{
public:
    cudaStream_t graph;
    cudaStream_t update;
    cudaStream_t back;
    gpu_stream()
    {
        std::cout << "Creating GPU streams..." << std::endl;
        cudaStreamCreate(&graph);
        cudaStreamCreate(&update);
        cudaStreamCreate(&back);
    }
};

#define WALKER_ID(walk) (walk.id)
#define WALKER_SOURCE(walk) (walk.source)
#define WALKER_PREVIOUS(walk) (walk.previous)
#define WALKER_POS(walk) (walk.current)
#define WALKER_HOP(walk) (walk.hop)
#define WALKER_CUR_BLOCK(walk) (walk.cur_index)
#define WALKER_PREV_BLOCK(walk) (walk.prev_index)

#endif
