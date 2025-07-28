#ifndef _GRAPH_CONFIG_H_
#define _GRAPH_CONFIG_H_

#include <string>
#include "api/types.hpp"

struct graph_config
{
    std::string base_name;
    size_t cache_size;
    size_t blocksize;
    tid_t nthreads;
    tid_t max_nthreads;
    vid_t nvertices;
    eid_t nedges;
    vid_t min_vert;
    bool is_weighted;
    wid_t numwalks;
    size_t walkpersource;
    size_t blockpergrid;
    size_t threadperblock;
    size_t cpu_threads;
    AlgorithmType algorithm;
    real_t p;
    real_t q;
    real_t alpha;
    hid_t maxhops;
    wid_t cpu_batch;
    wid_t gpu_batch;
    wid_t zero_threshold;
    bool gpu_schedule;
    bool cpu_schedule;
    bool walkaware;
};

#endif
