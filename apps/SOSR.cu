#include <omp.h>
#include <functional>
#include <iostream>
#include <fstream>
#include "api/types.hpp"
#include "api/constants.hpp"
#include "engine/config.hpp"
#include "engine/cache.hpp"
#include "engine/schedule.hpp"
#include "engine/walk.hpp"
#include "engine/engine.hpp"
#include "logger/logger.hpp"
#include "util/io.hpp"
#include "util/util.hpp"
#include "metrics/metrics.hpp"
#include "metrics/reporter.hpp"
#include "preprocess/graph_converter.hpp"
#include "preprocess/graph_sort.hpp"
#include "gpu/gpuwalk.hpp"
#include "gpu/memory.hpp"

int main(int argc, const char* argv[])
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 1)
    {
        cudaSetDevice(0);
        std::cout << "Using GPU 0" << std::endl;
    }
    assert(argc >= 2);
    set_argc(argc, argv);
    logstream(LOG_INFO) << "app : " << argv[0] << ", dataset : " << argv[1] << std::endl;
    std::string input = argv[1];
    bool weighted = get_option_bool("weighted");
    bool sorted = get_option_bool("sorted");
    bool skip = get_option_bool("skip");
    size_t blocksize_MB = get_option_long("blocksize", 1024);
    size_t memory_size = get_option_int("memory", MEMORY_CACHE / (1024LL * 1024 * 1024));
    wid_t walkpersource = (wid_t)get_option_int("walkpersource", 1);
    wid_t numwalks = (wid_t)get_option_int("numwalks", 0);
    hid_t steps = (hid_t)get_option_int("length", 30);
    real_t p = (real_t)get_option_float("p", 0.5);
    real_t q = (real_t)get_option_float("q", 2.0);
    real_t alpha = (real_t)get_option_float("alpha", 0.2);
    bool help_info = get_option_bool("h");
    size_t threadperblock = get_option_int("threadperblock", 512);
    size_t blockpergrid = get_option_int("blockpergrid", 512);
    size_t nthreads = get_option_int("nthreads", 70);
    size_t cpu_threads = get_option_int("cpu_threads", 32);
    wid_t walk_batch = get_option_int("walk_batch", 4096);
    wid_t zero_threshold = get_option_float("zero_threshold", 0);
    bool zero = get_option_bool("zero");
    bool gpu_schedule = get_option_bool("gpu");
    bool cpu_schedule = get_option_bool("cpu");
    bool walkaware = get_option_bool("walkaware");
    bool GOWalker = get_option_bool("GOWalker");
    bool SOWalker = get_option_bool("SOWalker");
    if (help_info)
    {
        std::cout << "- dataset:       the dataset path" << std::endl;
        std::cout << "- weighted:      whether the dataset is weighted" << std::endl;
        std::cout << "- sorted:        whether the vertex neighbors is sorted" << std::endl;
        std::cout << "- skip:          whether to skip preprocessing" << std::endl;
        std::cout << "- blocksize_MB:     the size of each block" << std::endl;
        std::cout << "- nthreads:      the number of threads to walk" << std::endl;
        std::cout << "- dynamic:       whether the blocksize is dynamic, according to the number of walks" << std::endl;
        std::cout << "- memory_size:   the size(GB) of memory" << std::endl;
        std::cout << "- max_iter:      the maximum number of iteration for simulated annealing scheduler" << std::endl;
        std::cout << "- walkpersource: the number of walks for each vertex" << std::endl;
        std::cout << "- length:        the number of step for each walk" << std::endl;
        std::cout << "- p:             node2vec parameter" << std::endl;
        std::cout << "- q:             node2vec parameter" << std::endl;
        std::cout << "- schedule:      scheduling strategy" << std::endl;
        std::cout << "- h:             print this message" << std::endl;
        return 0;
    }

    auto static_query_blocksize = [blocksize_MB](vid_t nvertices)
        { return blocksize_MB * (1024 * 1024); };
    std::function<size_t(vid_t nvertices)> query_blocksize;

    query_blocksize = static_query_blocksize;

    graph_converter converter(remove_extension(argv[1]), weighted, sorted);
    convert(input, converter, query_blocksize, skip);
    std::string base_name = remove_extension(argv[1]);
    /* graph meta info */
    vid_t nvertices;
    eid_t nedges;
    vid_t minimum_id;
    load_graph_meta(base_name, &nvertices, &nedges, &minimum_id, weighted);
    if (numwalks == 0)
    {
        numwalks = walkpersource * nvertices;
    }
    else
    {
        walkpersource = 0;
    }
    if (zero)
    {
        zero_threshold = blocksize_MB * (1024 * 1024) / 128 / 24;
        logstream(LOG_INFO) << "zero_threshold = " << zero_threshold << std::endl;
    }
    else
    {
        zero_threshold = 0;
    }
    graph_config conf = {
        base_name,
        memory_size * 1024LL * 1024 * 1024,
        blocksize_MB * (1024 * 1024),
        (tid_t)nthreads,
        (tid_t)omp_get_max_threads(),
        nvertices,
        nedges,
        minimum_id,
        weighted,
        numwalks,
        walkpersource,
        blockpergrid,
        threadperblock,
        cpu_threads,
        SOSR,
        p,
        q,
        alpha,
        steps,
        walk_batch,
        blockpergrid * threadperblock * walkperthread,
        zero_threshold,
        gpu_schedule,
        cpu_schedule,
        walkaware
    };

    metrics m("SOPR_numwalks_" + std::to_string(conf.numwalks) + "_steps_" + std::to_string(steps) + "_dataset_" + argv[1]);
    if (gpu_schedule)
    {
        m.set("gpu_schedule", 1);
    }
    else
    {
        m.set("gpu_schedule", 0);
    }
    if (cpu_schedule)
    {
        m.set("cpu_schedule", 1);
    }
    else
    {
        m.set("cpu_schedule", 0);
    }
    graph_block blocks(&conf);
    graph_driver driver(&conf, m);
    graph_walk walk_mangager(conf, driver, blocks);
    scheduler* walk_scheduler;
    if (GOWalker && !SOWalker) {
        walk_scheduler = new GOwalker_scheduler_t(m);
        logstream(LOG_INFO) << "Using GOWalker scheduler!" << std::endl;
    }
    else if (!GOWalker && SOWalker) {
        walk_scheduler = new SOwalker_scheduler_t(m);
        logstream(LOG_INFO) << "Using SOWalker scheduler!" << std::endl;
    }
    else {
        std::cout << "Please select one walker from GOWalker and SOWalker!" << std::endl;
        exit(0);
    }
    walk_scheduler = new GOwalker_scheduler_t(m);
    bid_t nmblocks = conf.cache_size / conf.blocksize;
    graph_cache cache(blocks.nblocks, blocksize_MB * (1024 * 1024));
    graph_cache g_cache(nmblocks, blocksize_MB * (1024 * 1024));
    graph_engine engine(cache, walk_mangager, driver, conf, m, g_cache, walk_scheduler);
    engine.prologue();
    engine.pipe_run();
    engine.epilogue();
    for (int i = 0; i < g_cache.ncblock; i++)
    {
        g_cache.cache_blocks[i].beg_pos = NULL;
        g_cache.cache_blocks[i].csr = NULL;
        g_cache.cache_blocks[i].weights = NULL;
        g_cache.cache_blocks[i].degree = NULL;
        g_cache.cache_blocks[i].block = NULL;
    }
    metrics_report(m);
    return 0;
}
