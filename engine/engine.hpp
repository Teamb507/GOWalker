#ifndef _GRAPH_ENGINE_H_
#define _GRAPH_ENGINE_H_

#include <functional>
#include <atomic>
#include <omp.h>
#include <iomanip>
#include "api/threadpool.hpp"
#include "api/types.hpp"
#include "cache.hpp"
#include "schedule.hpp"
#include "util/timer.hpp"
#include "metrics/metrics.hpp"
#include "engine/walk.hpp"
#include "gpu/memory.hpp"
#include "gpu/gpuwalk.hpp"
#include <ctime>
class graph_engine;
struct cpu_parameter
{
    tid_t tid;
    walker_t* walks;
    wid_t walknum;
    graph_engine* engine;
};
struct gpu_parameter
{
    bid_t blk;
    graph_engine* engine;
    scheduler* block_scheduler;
};
void walkjob(void* arg);

class graph_engine
{
public:
    graph_cache* cache;
    graph_walk* walk_manager;
    graph_driver* driver;
    graph_config* conf;
    scheduler* block_scheduler;
    graph_timer gtimer;
    std::vector<RandNum> seeds;
    metrics& _m;
    graph_cache* h_cache;
    graph_test* test;
    gpu_stream* stream;
    cache_block* d_map;
    gpu_graph* g_graph;
    gpu_cache* g_cache;
    curandState* states;
    gpu_walks** g_walks;
    ThreadPool* cpu_updatepool;
    graph_engine(graph_cache& _cache, graph_walk& manager, graph_driver& _driver, graph_config& _conf, metrics& m, graph_cache& _g_cache, scheduler* shcedule) : _m(m)
    {
        cache = &_cache;
        walk_manager = &manager;
        driver = &_driver;
        conf = &_conf;
        block_scheduler = shcedule;
        h_cache = &_g_cache;
        test = new graph_test();
        stream = new gpu_stream();
#ifdef NO_pipeline
        stream->update = stream->graph;
        stream->back = stream->graph;
#endif
        seeds = std::vector<RandNum>(conf->nthreads, RandNum(9898676785859));
        cpu_updatepool = new ThreadPool(conf->cpu_threads, conf->cpu_threads);
        wid_t gpu_max_walks = conf->threadperblock * conf->blockpergrid * walkperthread;
        logstream(LOG_DEBUG) << "Graph blocks : " << walk_manager->global_blocks->nblocks << ", GPU memory blocks : " << h_cache->ncblock << std::endl;
        _m.set("Graph blocks", static_cast<int>(walk_manager->global_blocks->nblocks));
        _m.set("Gpu memory blocks", static_cast<int>(h_cache->ncblock));
        for (int i = 0; i < cache->ncblock; i++)
        {
            driver->load_block_info_all(*cache, walk_manager->global_blocks, i, i, conf->blocksize);
        }
        logstream(LOG_INFO) << "Graph loading is complete!" << std::endl;
        g_graph = initgpugraph(walk_manager->global_blocks, stream->graph);
        g_cache = initgpucache(cache, walk_manager->global_blocks, h_cache, driver->_weighted, conf->blocksize, stream->graph);
        states = cudarand(conf->blockpergrid, conf->threadperblock, stream->graph);
        g_walks = (gpu_walks**)malloc(sizeof(gpu_walks*) * 2);
        g_walks[0] = initwalk(gpu_max_walks, conf->maxhops, walk_manager->totblocks, stream->graph);
        g_walks[1] = initwalk(gpu_max_walks, conf->maxhops, walk_manager->totblocks, stream->graph);
        cudaHostAlloc((void**)&d_map, sizeof(cache_block) * walk_manager->global_blocks->nblocks, cudaHostAllocMapped);
        for (int i = 0; i < walk_manager->global_blocks->nblocks; i++)
        {
            d_map[i].block = NULL;
            d_map[i].beg_pos = cache->cache_blocks[i].beg_pos;
            d_map[i].csr = cache->cache_blocks[i].csr;
            d_map[i].weights = cache->cache_blocks[i].weights;
        }
        if (cpu_updatepool == NULL)
        {
            std::cout << "cpu thread pool create failed!" << std::endl;
            exit(0);
        }
        for (tid_t tid = 0; tid < conf->nthreads; tid++)
        {
            seeds[tid] = time(NULL) + tid;
        }
    }
    ~graph_engine()
    {
        delete test;
    }

    void copy(graph_engine* engine)
    {
        engine->cache = cache;
        engine->walk_manager = walk_manager;
    }
    void prologue()
    {
        logstream(LOG_INFO) << "  =================  STARTED  ======================  " << std::endl;
        logstream(LOG_INFO) << "Random walks, random generate " << conf->numwalks << " walks on whole graph, exec_threads = " << conf->nthreads << std::endl;
        logstream(LOG_INFO) << "vertices : " << conf->nvertices << ", edges : " << conf->nedges << std::endl;
        srand(time(0));
        walk_manager->gpu_createwalk(conf->walkpersource, conf->numwalks, conf, stream->graph);
        logstream(LOG_INFO) << "Initial walks are complete!" << std::endl;
        return;
    }

    void epilogue()
    {
#ifdef TEST
        test->aggregate_gpu();
        uint64_t totalsteps = test->gpu_in_steps + test->gpu_pass_steps + test->cpu_allsteps;
        logstream(LOG_DEBUG) << "1. GPU steps: " << test->gpu_in_steps + test->gpu_pass_steps << " ("
            << std::fixed << std::setprecision(2)
            << (totalsteps == 0 ? 0.0 : static_cast<double>(test->gpu_in_steps + test->gpu_pass_steps) / totalsteps * 100)
            << "%)" << std::endl;
        logstream(LOG_DEBUG) << "1.1 GPU in steps: " << test->gpu_in_steps << " ("
            << std::fixed << std::setprecision(2)
            << (totalsteps == 0 ? 0.0 : static_cast<double>(test->gpu_in_steps) / totalsteps * 100)
            << "%)" << std::endl;
        logstream(LOG_DEBUG) << "1.2 GPU pass steps: " << test->gpu_pass_steps << " ("
            << std::fixed << std::setprecision(2)
            << (totalsteps == 0 ? 0.0 : static_cast<double>(test->gpu_pass_steps) / totalsteps * 100)
            << "%)" << std::endl;
        logstream(LOG_DEBUG) << "2. CPU steps: " << test->cpu_allsteps << " ("
            << std::fixed << std::setprecision(2)
            << (totalsteps == 0 ? 0.0 : static_cast<double>(test->cpu_allsteps) / totalsteps * 100)
            << "%)" << std::endl;
        logstream(LOG_DEBUG) << "Total steps: " << totalsteps << std::endl;
        logstream(LOG_DEBUG) << "-------IO utilization rate-------- " << std::endl;
        logstream(LOG_DEBUG) << "seq    edgenum     updatesteps    %" << std::endl;
        std::ofstream ofs("IO_uti.txt");
        for (int i = 0; i < test->IO_uti.size(); i++)
        {
            logstream(LOG_DEBUG) << i << "    " << test->IO_uti[i].first << "  " << test->IO_uti[i].second << "  " << ((double)test->IO_uti[i].second) / ((double)test->IO_uti[i].first) * 100.0 << std::endl;
            ofs << i << " " << test->IO_uti[i].first << " " << test->IO_uti[i].second << " " << ((double)test->IO_uti[i].second) / ((double)test->IO_uti[i].first) * 100.0 << "\n";
        }
#endif
        logstream(LOG_INFO) << "  ================= FINISHED ======================  " << std::endl;
    }

    int sidewalkupdate(walker_t& walker, real_t p, real_t q, hid_t maxhop, RandNum* seed, graph_engine* cpu_engine, tid_t tid)
    {
        bid_t nblocks = walk_manager->global_blocks->nblocks;
        vid_t cur_vertex = walker.current, prev_vertex = walker.previous;
        if (cur_vertex != walker.current)
        {
            return 0;
        }
        hid_t hop = walker.hop;
        bid_t cur_blk = walker.cur_index, prev_blk = walker.prev_index;
        int run_step = 0;
        real_t inv_p = 1.0f / p;
        real_t inv_q = 1.0f / q;
        real_t max_val = std::max(inv_p, std::max(inv_q, 1.0f));
        real_t min_val = std::min(inv_p, std::min(inv_q, 1.0f));
        while (__builtin_expect(cur_blk != prev_blk && hop < maxhop, 0))
        {
            cache_block* cur_block = &(cache->cache_blocks[cur_blk]);
            cache_block* prev_block = &(cache->cache_blocks[prev_blk]);
            vid_t cur_block_start = cur_block->block->start_vert;
            vid_t cur_block_end = cur_block_start + cur_block->block->nverts;
            vid_t prev_block_start = prev_block->block->start_vert;

            eid_t* cur_beg_pos = cur_block->beg_pos;
            eid_t* prev_beg_pos = prev_block->beg_pos;
            vid_t* cur_csr = cur_block->csr;
            vid_t* prev_csr = prev_block->csr;

            vid_t off = cur_vertex - cur_block_start;
            vid_t prev_off = prev_vertex - prev_block_start;

            eid_t adj_head = cur_beg_pos[off] - cur_block->block->start_edge;
            eid_t adj_tail = cur_beg_pos[off + 1] - cur_block->block->start_edge;
            eid_t prev_adj_head = prev_beg_pos[prev_off] - prev_block->block->start_edge;
            eid_t prev_adj_tail = prev_beg_pos[prev_off + 1] - prev_block->block->start_edge;

            vid_t next_vertex = 0;
            eid_t deg = adj_tail - adj_head;
            if (__builtin_expect(deg == 0, 0))
            {
                hop = maxhop - 1;
            }
            else
            {
                bool accept = false;
                size_t rand_pos = 0;
                while (!accept)
                {
                    real_t rand_val;
                    seed->inlineRandPair(max_val, deg, rand_val, rand_pos);
                    vid_t candidate = cur_csr[adj_head + rand_pos];
                    if (rand_val <= min_val)
                    {
                        accept = true;
                        break;
                    }
                    if (candidate == prev_vertex)
                    {
                        if (rand_val < inv_p)
                        {
                            accept = true;
                            break;
                        }
                    }
                    else
                    {
                        bool found = false;
                        int64_t left = (int64_t)prev_adj_head, right = (int64_t)prev_adj_tail - 1;
                        while (left <= right)
                        {
                            int64_t mid = left + ((right - left) >> 1);
                            vid_t mid_val = prev_csr[mid];
                            if (mid_val == candidate)
                            {
                                found = true;
                                break;
                            }
                            else if (mid_val < candidate)
                            {
                                left = mid + 1;
                            }
                            else
                            {
                                right = mid - 1;
                            }
                        }

                        if (found)
                        {
                            if (rand_val < 1.0f)
                            {
                                accept = true;
                            }
                        }
                        else
                        {
                            if (rand_val < inv_q)
                            {
                                accept = true;
                            }
                        }
                    }
                }
                next_vertex = cur_csr[adj_head + rand_pos];
            }
            prev_vertex = cur_vertex;
            cur_vertex = next_vertex;
            prev_blk = cur_blk;
            hop++;
            run_step++;
            if (__builtin_expect(!(cur_vertex >= cur_block_start && cur_vertex < cur_block_end), 0))
            {
                cur_blk = walk_manager->global_blocks->get_block(cur_vertex);
            }
            else
                break;
        }
        if (hop < maxhop)
        {
            bid_t blk = prev_blk * nblocks + cur_blk;
            int res = cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].push(prev_vertex, cur_vertex, prev_blk, cur_blk, hop);
            if (res == 2)
            {
                cpu_engine->walk_manager->cpuinsertbatchwalk(cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].array, cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].bsize, blk);
                cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].emptyarray();
                cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].alloc();
            }
        }
        return run_step;
    }

    int SOPR_sidewalkupdate(walker_t& walker, real_t alpha, hid_t maxhop, RandNum* seed, graph_engine* cpu_engine, tid_t tid)
    {
        bid_t nblocks = walk_manager->global_blocks->nblocks;
        vid_t cur_vertex = walker.current, prev_vertex = walker.previous;
        hid_t hop = walker.hop;
        bid_t cur_blk = walker.cur_index, prev_blk = walker.prev_index;
        int run_step = 0;
        while (__builtin_expect(cur_blk != prev_blk && hop < maxhop, 1))
        {
            cache_block* cur_block = &(cache->cache_blocks[cur_blk]);
            cache_block* prev_block = &(cache->cache_blocks[prev_blk]);
            vid_t cur_block_start = cur_block->block->start_vert;
            vid_t cur_block_end = cur_block_start + cur_block->block->nverts;
            vid_t prev_block_start = prev_block->block->start_vert;
            eid_t* cur_beg_pos = cur_block->beg_pos;
            eid_t* prev_beg_pos = prev_block->beg_pos;
            vid_t* cur_csr = cur_block->csr;
            vid_t* prev_csr = prev_block->csr;
            vid_t off = cur_vertex - cur_block_start;
            vid_t prev_off = prev_vertex - prev_block_start;
            eid_t adj_head = cur_beg_pos[off] - cur_block->block->start_edge;
            eid_t adj_tail = cur_beg_pos[off + 1] - cur_block->block->start_edge;
            eid_t prev_adj_head = prev_beg_pos[prev_off] - prev_block->block->start_edge;
            eid_t prev_adj_tail = prev_beg_pos[prev_off + 1] - prev_block->block->start_edge;

            vid_t next_vertex = 0;
            eid_t deg = adj_tail - adj_head;
            eid_t prev_deg = prev_adj_tail - prev_adj_head;
            if (__builtin_expect(deg == 0, 0))
            {
                hop = maxhop - 1;
            }
            else
            {
                real_t inv_deg = (1.0 - alpha) / deg;
                real_t inv_prev_deg = alpha / prev_deg;
                real_t max_val = std::max(inv_deg, inv_deg + inv_prev_deg);
                real_t min_val = std::min(inv_deg, inv_deg + inv_prev_deg);
                bool accept = false;
                size_t rand_pos = 0;
                while (!accept)
                {
                    real_t rand_val;
                    seed->inlineRandPair(max_val, deg, rand_val, rand_pos);
                    vid_t candidate = cur_csr[adj_head + rand_pos];
                    if (rand_val <= min_val)
                    {
                        accept = true;
                        break;
                    }
                    if (candidate == prev_vertex)
                    {
                        if (rand_val < (1.0 - alpha) / deg)
                            accept = true;
                    }
                    else
                    {
                        bool found = false;
                        int64_t left = (int64_t)prev_adj_head, right = (int64_t)prev_adj_tail - 1;
                        while (left <= right)
                        {
                            int64_t mid = left + ((right - left) >> 1);
                            vid_t mid_val = prev_csr[mid];
                            if (mid_val == candidate)
                            {
                                found = true;
                                break;
                            }
                            else if (mid_val < candidate)
                            {
                                left = mid + 1;
                            }
                            else
                            {
                                right = mid - 1;
                            }
                        }
                        if (found)
                        {
                            if (rand_val < (inv_deg + inv_prev_deg))
                                accept = true;
                        }
                        else
                        {
                            if (rand_val < inv_deg)
                                accept = true;
                        }
                    }
                }
                next_vertex = cur_csr[adj_head + rand_pos];
            }
            prev_vertex = cur_vertex;
            cur_vertex = next_vertex;
            prev_blk = cur_blk;
            hop++;
            run_step++;
            if (__builtin_expect(!(cur_vertex >= cur_block_start && cur_vertex < cur_block_end), 0))
            {
                cur_blk = walk_manager->global_blocks->get_block(cur_vertex);
            }
            else
                break;
        }
        if (hop < maxhop)
        {
            bid_t blk = prev_blk * nblocks + cur_blk;
            int res = cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].push(prev_vertex, cur_vertex, prev_blk, cur_blk, hop);
            if (res == 2)
            {
                cpu_engine->walk_manager->cpuinsertbatchwalk(cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].array, cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].bsize, blk);
                cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].emptyarray();
                cpu_engine->walk_manager->cpu_walkbuffer[blk][tid].alloc();
            }
        }
        return run_step;
    }

    void pipeline()
    {
        _m.start_time("1.1_transfer");
        eid_t edgenum = 0;
        int blknums = block_scheduler->cachesize();
        for (int pos = 0; pos < block_scheduler->cachesize(); pos++)
        {
            edgenum += block_scheduler->copyblocks(*h_cache, *cache, *driver, *walk_manager, g_cache, g_graph, pos, stream->graph);
        }
        checkCudaError(cudaStreamSynchronize(stream->graph));
        _m.stop_time("1.1_transfer");
        wid_t offset[2] = { 0,0 };
        for (int i = 0; i >= 0; i++)
        {
            _m.start_time("1.1_transfer");
            wid_t nwalks = walk_manager->copywalker(g_walks[i % 2], conf->gpu_batch, offset[i % 2], h_cache, conf, stream->graph, _m);
            _m.stop_time("1.1_transfer");
            if (nwalks <= 0)
            {
                checkCudaError(cudaStreamSynchronize(stream->update));//update
                _m.start_time("1.1_transfer");
                offset[(i + 1) % 2] = walk_manager->copy_back(g_walks[(i + 1) % 2], h_cache, stream->back, _m);//copyback
                _m.stop_time("1.1_transfer");
                break;
            }
            if (i > 0)
            {
                checkCudaError(cudaStreamSynchronize(stream->update));//update
                _m.start_time("1.1_transfer");
                offset[(i + 1) % 2] = walk_manager->copy_back(g_walks[(i + 1) % 2], h_cache, stream->back, _m);//copyback
                _m.stop_time("1.1_transfer");
            }
            checkCudaError(cudaStreamSynchronize(stream->graph));//copy
            _m.start_time("1.3_GPU_update");
            if (conf->algorithm == SOPR)
                SOPR_gpu_run << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[i % 2], g_cache, g_graph, conf->alpha, states, test->d_gpu);
            else
                gpu_run << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[i % 2], g_cache, g_graph, conf->p, conf->q, states, test->d_gpu);
            prefix << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[i % 2], g_graph);
            insertglobal << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[i % 2], g_cache, g_graph);
            walkswap << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[i % 2]);
            _m.stop_time("1.3_GPU_update");
        }
        checkCudaError(cudaStreamSynchronize(stream->graph));
        checkCudaError(cudaStreamSynchronize(stream->back));
        checkCudaError(cudaStreamSynchronize(stream->update));

#ifdef TEST
        test->utilization_rate(edgenum);
#endif

    }

    void zero_gpujob(bid_t blk)
    {
        wid_t nwalks = walk_manager->zerocopywalktogpu(blk);
        wid_t gpu_load = 0;
        wid_t gpu_max_walks = conf->gpu_batch;
        int i = 0;
        while (walk_manager->g_walks != NULL)
        {
            if (i % 2 == 1)
            {
                cudaStreamSynchronize(stream->update);
                while (gpu_load < gpu_max_walks)
                {
                    wid_t batch_walks = walk_manager->g_walks->bsize;
                    if (gpu_load + batch_walks > gpu_max_walks)
                        break;
                    copywalktogpu(g_walks[1], walk_manager->g_walks->array, batch_walks, conf->maxhops, gpu_load, walk_manager->nblocks, stream->update);
                    gpu_load += batch_walks;
                    auto p = walk_manager->g_walks;
                    walk_manager->g_walks = walk_manager->g_walks->next;
                    checkCudaError(cudaFreeHost(p->array));
                    free(p);
                    if (walk_manager->g_walks == NULL)
                        break;
                }
                if (conf->algorithm == SOPR)
                    SOPR_zerocopy_update << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[1], d_map, g_graph, conf->alpha, walk_manager->nblocks, states, test->d_gpu);
                else
                    zerocopy_update << <conf->blockpergrid, conf->threadperblock, 0, stream->update >> > (g_walks[1], d_map, g_graph, conf->p, conf->q, walk_manager->nblocks, states, test->d_gpu);
                gpu_load = 0;
            }
            else
            {
                cudaStreamSynchronize(stream->back);
                while (gpu_load < gpu_max_walks)
                {
                    wid_t batch_walks = walk_manager->g_walks->bsize;
                    if (gpu_load + batch_walks > gpu_max_walks)
                        break;
                    copywalktogpu(g_walks[0], walk_manager->g_walks->array, batch_walks, conf->maxhops, gpu_load, walk_manager->nblocks, stream->update);
                    gpu_load += batch_walks;
                    auto p = walk_manager->g_walks;
                    walk_manager->g_walks = walk_manager->g_walks->next;
                    checkCudaError(cudaFreeHost(p->array));
                    free(p);
                    if (walk_manager->g_walks == NULL)
                        break;
                }
                if (conf->algorithm == SOPR)
                    SOPR_zerocopy_update << <conf->blockpergrid, conf->threadperblock, 0, stream->back >> > (g_walks[0], d_map, g_graph, conf->alpha, walk_manager->nblocks, states, test->d_gpu);
                else
                    zerocopy_update << <conf->blockpergrid, conf->threadperblock, 0, stream->back >> > (g_walks[0], d_map, g_graph, conf->p, conf->q, walk_manager->nblocks, states, test->d_gpu);
                gpu_load = 0;
            }
            i++;
        }
    }

    void pipe_run()
    {
        while (walk_manager->nwalks() > 0)
        {
            std::atomic<bool> gpu_working(true);
            wid_t load = 0;
            bid_t nblocks = walk_manager->nblocks;
            for (bid_t i = 0; i < walk_manager->totblocks; i++)
            {
                for (int tid = 0; tid < conf->cpu_threads; tid++)
                {
                    if (walk_manager->cpu_walkbuffer[i][tid].bsize > 0)
                    {
                        load += walk_manager->cpuinsertbatchwalk(walk_manager->cpu_walkbuffer[i][tid].array, walk_manager->cpu_walkbuffer[i][tid].bsize, i);
                        walk_manager->cpu_walkbuffer[i][tid].emptyarray();
                        walk_manager->cpu_walkbuffer[i][tid].alloc();
                    }
                }
            }
            _m.start_time("0_Total_time");

            bool res = block_scheduler->schedule(*conf, *h_cache, *driver, *walk_manager, g_cache, g_graph, *cache, 0, _m, stream->graph);

            if (res)
            {

#pragma omp parallel sections
                {
#pragma omp section
                    {
                        _m.start_time("1_GPUTime");
                        pipeline();
                        gpu_working.store(false);
                        _m.stop_time("1_GPUTime");
                    }
#pragma omp section
                    {
                        _m.start_time("2_CPUTime");
                        bid_t blk = 0;
                        for (int i = 0;i < block_scheduler->cpu_blk.size();i++)
                        {
                            if (gpu_working.load())
                            {
                                blk = block_scheduler->cpu_blk[i];
                            }
                            else
                            {
                                break;
                            }
#ifdef IO_UTE
                            std::cout << "cpu load blk:" << blk << std::endl;
#endif
                            while (walk_manager->cpu_loadblkwalk(blk) > 0)
                            {
                                walker_t* walks = walk_manager->c_walks->array;
                                cpu_parameter* arg = (cpu_parameter*)malloc(sizeof(cpu_parameter));
                                *arg = { 0, walks, (wid_t)walk_manager->c_walks->bsize, this };
                                while (cpu_updatepool->pushJob(walkjob, arg, sizeof(cpu_parameter)) != 1)
                                    ;
                                auto p = walk_manager->c_walks;
                                walk_manager->c_walks = NULL;
                                free(p);
                                if (!gpu_working.load())
                                {
                                    break;
                                }
                            }
                        }
                        cpu_updatepool->wait();
#ifdef IO_UTE
                        std::cout << "cpu load over" << std::endl;
#endif
                        _m.stop_time("2_CPUTime");
                    }
                }
            }
            else
            {
                wid_t load = 0;
                bid_t nblocks = walk_manager->nblocks;
                for (bid_t i = 0; i < walk_manager->totblocks; i++)
                {
                    for (int tid = 0; tid < conf->cpu_threads; tid++)
                    {
                        if (walk_manager->cpu_walkbuffer[i][tid].bsize > 0)
                        {
                            load += walk_manager->cpuinsertbatchwalk(walk_manager->cpu_walkbuffer[i][tid].array, walk_manager->cpu_walkbuffer[i][tid].bsize, i);
                            walk_manager->cpu_walkbuffer[i][tid].emptyarray();
                            walk_manager->cpu_walkbuffer[i][tid].alloc();
                        }
                    }
                }
#ifdef IO_UTE
                std::cout << "收集cpu buffer walks:" << load << std::endl;
                std::cout << "walks:" << walk_manager->nwalks() << std::endl;
                for (int i = 0; i < walk_manager->global_blocks->nblocks; i++)
                {
                    std::cout << "blk:" << i << "load :";
                    for (int j = 0; j < walk_manager->global_blocks->nblocks; j++)
                    {
                        bid_t blk = i * walk_manager->global_blocks->nblocks + j;
                        std::cout << " " << walk_manager->nblockwalks(blk);
                    }
                    std::cout << std::endl;
                }
#endif
                _m.start_time("3_Zero_Time");
                for (bid_t i = 0; i < walk_manager->totblocks; i++)
                {
                    if (walk_manager->block_walks[i].block_numwalks() > 0)
                    {
                        gpu_parameter* arg = (gpu_parameter*)malloc(sizeof(gpu_parameter));
                        *arg = { i, this, NULL };
                        zero_gpujob(i);
                    }
                }
                _m.stop_time("3_Zero_Time");
            }

            _m.stop_time("0_Total_time");
        }
    }
};
void walkjob(void* arg)
{
    cpu_parameter* para = (cpu_parameter*)arg;
    walker_t* walks = para->walks;
    wid_t walknum = para->walknum;
    graph_engine* engine = para->engine;
    graph_config* conf = engine->conf;
    tid_t tid = para->tid;
    uint64_t steps = 0;
    real_t p = engine->conf->p;
    real_t q = engine->conf->q;
    real_t alpha = engine->conf->alpha;
    hid_t maxhop = engine->conf->maxhops;
    RandNum* seed = &engine->seeds[tid];
    std::vector<bid_t> buckets = engine->block_scheduler->buckets;
    if (engine->conf->algorithm == node2vec)
    {
        for (wid_t i = 0; i < walknum; i++)
        {
            steps += (uint64_t)engine->sidewalkupdate(walks[i], p, q, maxhop, seed, engine, tid);
        }
    }
    else if (engine->conf->algorithm == SOPR)
    {
        for (wid_t i = 0; i < walknum; i++)
        {
            steps += (uint64_t)engine->SOPR_sidewalkupdate(walks[i], alpha, maxhop, seed, engine, tid);
        }
    }
    checkCudaError(cudaFreeHost(walks));
#ifdef TEST
    __sync_fetch_and_add(&engine->test->cpu_allsteps, steps);
#endif
}

#endif
