#ifndef _GRAPH_SCHEDULE_H_
#define _GRAPH_SCHEDULE_H_
#include "cache.hpp"
#include "config.hpp"
#include "driver.hpp"
#include "gpu/memory.hpp"
#include "metrics/metrics.hpp"
#include "metrics/reporter.hpp"
#include "util/io.hpp"
#include "util/util.hpp"
#include "walk.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#define TEMPERATURE_COOLING
template <typename value_t> struct rank_compare {
  bool operator()(const std::pair<bid_t, value_t>& p1,
    const std::pair<bid_t, value_t>& p2) {
    return p1.second < p2.second;
  }
};

/** graph_scheduler
 *
 * This file contribute to define the interface how to schedule cache blocks
 */

class scheduler {
protected:
  metrics& _m;

public:
  scheduler(metrics& m) : _m(m) {}
  std::vector<bid_t> buckets; // 加载进入gpu中的块
  std::vector<bid_t> cpu_blk;
  virtual bool schedule(graph_config& conf, graph_cache& cache,
    graph_driver& driver, graph_walk& walk_manager,
    gpu_cache* g_cache, gpu_graph* g_graph,
    graph_cache& c_cache, cudaStream_t cstream, metrics& _m,
    cudaStream_t copy) {
    return;
  }
  virtual eid_t copyblocks(graph_cache& cache, graph_cache& c_cache,
    graph_driver& driver, graph_walk& walk_manager,
    gpu_cache* g_cache, gpu_graph* g_graph, int pos,
    cudaStream_t copy) {
    return;
  }
  virtual bid_t posblk(int pos) { return 0; }
  virtual bid_t cachesize() { return 0; }
  ~scheduler() {}
};

// cpu-gpu hybrid
class GOwalker_scheduler_t : public scheduler {
public:
  std::vector<bid_t> bucket_sequences;
  size_t scheduleblocks = 0;
  GOwalker_scheduler_t( metrics& m) : scheduler(m) {
  }

  GOwalker_scheduler_t(std::vector<bid_t> copybuckets, metrics& m): scheduler(m) {
    buckets.resize(copybuckets.size());
    std::copy(copybuckets.begin(), copybuckets.end(), buckets.begin());
  }

  int Combination(int n, int k) {
    if (k < 0 || k > n) {
      return 0;
    }
    int res = 1;
    for (int i = 0; i < k; ++i) {
      res *= (n - i);
      res /= (i + 1);
    }
    return res;
  }
  bool max_choose_blocks(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    _m.start_time("max_choose_blocks");
    bid_t nblocks = walk_manager.nblocks;
    bid_t totblk = walk_manager.totblocks;
    std::vector<bid_t> cached = buckets;
    std::vector<bid_t> pre_cached;
    for (int i = 0;i < nblocks;i++)
    {
      if (std::find(cached.begin(), cached.end(), i) == cached.end())
      {
        pre_cached.push_back(i);
      }
    }
    if (cached.size() >= cache.ncblock)
    {
      std::vector<std::pair<bid_t, size_t>> block_score;
      for (int i = 0;i < pre_cached.size();i++)
      {
        size_t min=0xffffffff;
        bid_t index=0;
        size_t all=0;
        for (int j = 0;j < cached.size();j++)
        {
          size_t score = walk_manager.block_walks[pre_cached[i] * nblocks + cached[j]].block_numwalks() + walk_manager.block_walks[cached[j] * nblocks + pre_cached[i]].block_numwalks();
          all+=score;
          if(score<min){
            index=j;
            min=score;
          }
        }
        block_score.push_back(std::make_pair(pre_cached[i] * nblocks + index, all-min));
      }
      std::sort(block_score.begin(), block_score.end(), [](const std::pair<bid_t, size_t>& a, const std::pair<bid_t, size_t>& b) { return a.second > b.second; });
      if (block_score[0].second > 0 && block_score[0].second > conf.zero_threshold)
      {
        bid_t choose_blk = block_score[0].first / nblocks;
        bid_t prev_index = block_score[0].first % nblocks;
        buckets[prev_index]=choose_blk;
        _m.stop_time("max_choose_blocks");
        return true;
      }
    }
    else
    {
      std::vector<std::pair<bid_t, size_t>> block_score;
      //bid_t记录为要选择的块
      for (int i = 0;i < pre_cached.size();i++)
      {
        bid_t pre_choose = pre_cached[i];
        size_t score = 0;
        for (int j = 0;j < cached.size();j++)
        {
          score += walk_manager.block_walks[pre_choose * nblocks + cached[j]].block_numwalks() + walk_manager.block_walks[cached[j] * nblocks + pre_choose].block_numwalks();
        }
        score += walk_manager.block_walks[pre_choose * nblocks + pre_choose].block_numwalks();
        block_score.push_back(std::make_pair(pre_choose, score));
      }
      std::sort(block_score.begin(), block_score.end(), [](const std::pair<bid_t, size_t>& a, const std::pair<bid_t, size_t>& b) { return a.second > b.second; });
      if (block_score[0].second > 0 && block_score[0].second > conf.zero_threshold)
      {
        buckets.push_back(block_score[0].first);
        _m.stop_time("max_choose_blocks");
        return true;
      }
    }

    _m.stop_time("max_choose_blocks");
    return false;
  }

  bid_t posblk(int pos) { return buckets[pos]; }

  bid_t cachesize() { return buckets.size(); }

  eid_t copyblocks(graph_cache& cache, graph_cache& c_cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, int pos, cudaStream_t copy) {
    eid_t load = 0;
    cache.walk_blocks.clear();
    std::vector<bid_t> cached;
    bid_t choosed = buckets[pos];
    for (int i = 0; i <= pos; i++) {
      for (int j = 0; j <= pos; j++) {
        bid_t blk = buckets[i] * c_cache.ncblock + buckets[j];
        cache.walk_blocks.push_back(blk);
      }
    }
    // 看是否有空的
    for (int i = 0; i < cache.ncblock; i++) {
      if (cache.cache_blocks[i].block == NULL) {
        // 加载新的块
        cache.cache_blocks[i].block = c_cache.cache_blocks[choosed].block;
        cache.cache_blocks[i].block->cache_index = i;
        cache.cache_blocks[i].block->status = ACTIVE;
        cache.cache_blocks[i].beg_pos = c_cache.cache_blocks[choosed].beg_pos;
        cache.cache_blocks[i].csr = c_cache.cache_blocks[choosed].csr;
        if (driver._weighted)
          cache.cache_blocks[i].weights = c_cache.cache_blocks[choosed].weights;
        copycachetogpu(g_cache, &cache, i, driver._weighted, copy);
        pipeline_gpugraph(g_graph, cache.cache_blocks[i].block, copy);
        load += cache.cache_blocks[i].block->nedges;
        return load;
      }
      else if (cache.cache_blocks[i].block->blk == choosed) {
        // 如果块已经再显存中不要任何操作；
        return load;
      }
      else {
        cached.push_back(cache.cache_blocks[i].block->blk);
      }
    }
    // 替换不在cache中的
    for (int i = 0; i < cache.ncblock; i++) {
      if (std::find(buckets.begin(), buckets.end(),
        cache.cache_blocks[i].block->blk) ==
        buckets.end()) // 找到cache里面没有被选中的
      {
        // 无效旧块
        cache.cache_blocks[i].block->cache_index = c_cache.ncblock;
        cache.cache_blocks[i].block->status = INACTIVE;
        pipeline_gpugraph(g_graph, cache.cache_blocks[i].block, copy);
        // 加载新的块
        cache.cache_blocks[i].block = c_cache.cache_blocks[choosed].block;
        cache.cache_blocks[i].block->cache_index = i;
        cache.cache_blocks[i].block->status = ACTIVE;
        cache.cache_blocks[i].beg_pos = c_cache.cache_blocks[choosed].beg_pos;
        cache.cache_blocks[i].csr = c_cache.cache_blocks[choosed].csr;
        if (driver._weighted)
          cache.cache_blocks[i].weights = c_cache.cache_blocks[choosed].weights;
        copycachetogpu(g_cache, &cache, i, driver._weighted, copy);
        pipeline_gpugraph(g_graph, cache.cache_blocks[i].block, copy);
        load += cache.cache_blocks[i].block->nedges;
        return load;
      }
    }
  }

  bool cpu_firstcol(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    cpu_blk.clear();
    for (int i = 0;i < buckets.size();i++)
    {
      bid_t t = buckets[i];
      for (int f = 0;f < walk_manager.nblocks;f++)
      {
        bid_t totblk = f * walk_manager.nblocks + t;
        if (std::find(buckets.begin(), buckets.end(), f) != buckets.end())
          continue;
        if (walk_manager.nblockwalks(totblk) > 0)
        {
          cpu_blk.push_back(totblk);
        }
      }
    }
    return true;
  }

  bool cpu_firstrow(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    cpu_blk.clear();
    for (int f = 0;f < walk_manager.nblocks;f++)
    {
      for (int t = 0;t < walk_manager.nblocks;t++)
      {
        if (f == t)
          continue;
        if (std::find(buckets.begin(), buckets.end(), f) != buckets.end() && std::find(buckets.begin(), buckets.end(), t) != buckets.end())
        {
          continue; // 这两个块都在gpu上
        }
        bid_t totblk = f * walk_manager.nblocks + t;
        if (walk_manager.nblockwalks(totblk) > 0)
        {
          cpu_blk.push_back(totblk);
        }
      }
    }
    return true;
  }

  bool basic_choose(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    static bid_t choose = 0;
    if (buckets.size() < cache.ncblock)
    {
      buckets.push_back(choose);
      choose = (choose + 1) % walk_manager.nblocks;
      return true;
    }
    else {
      bid_t over = rand() % cache.ncblock;
      buckets[over] = choose;
      choose = (choose + 1) % walk_manager.nblocks;
      return true;
    }
  }

  bool walkaware(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    _m.start_time("walkaware_choose_blocks");
    bid_t nblocks = walk_manager.nblocks;
    bid_t totblk = walk_manager.totblocks;
    std::vector<bid_t> cached = buckets;
    std::vector<bid_t> pre_cached;
    for (int i = 0;i < nblocks;i++)
    {
      if (std::find(cached.begin(), cached.end(), i) == cached.end())
      {
        pre_cached.push_back(i);
      }
    }
    if (cached.size() >= cache.ncblock)
    {
      std::vector<std::pair<bid_t, size_t>> block_score;
      //bid_t 记录为pre_cahced*nblcoks+cached为使用pre
      for (int i = 0;i < pre_cached.size();i++)
      {
        for (int j = 0;j < cached.size();j++)
        {
          bid_t index = pre_cached[i] * nblocks + cached[j];
          size_t score = 0;
          for (int k = 0;k < cached.size();k++)
          {
            if (j != k)
            {
              bid_t blk1 = pre_cached[i] * nblocks + cached[k];
              bid_t blk2 = cached[k] * nblocks + pre_cached[i];
              score += walk_manager.block_walks[blk1].block_numwalks() + walk_manager.block_walks[blk2].block_numwalks();
            }
          }
          score += walk_manager.block_walks[pre_cached[i] * nblocks + pre_cached[i]].block_numwalks();
          block_score.push_back(std::make_pair(index, score));
        }
      }
      std::sort(block_score.begin(), block_score.end(), [](const std::pair<bid_t, size_t>& a, const std::pair<bid_t, size_t>& b) { return a.second > b.second; });
      if (block_score[0].second > 0 && block_score[0].second > conf.zero_threshold)
      {
        bid_t choose = block_score[0].first / nblocks;
        bid_t prev_blk = block_score[0].first % nblocks;
        for (int i = 0;i < buckets.size();i++)
        {
          if (buckets[i] == prev_blk)
          {
            buckets[i] = choose;
            _m.stop_time("walkaware_choose_blocks");
            return true;
          }
        }
      }
    }
    else
    {
      std::vector<std::pair<bid_t, size_t>> block_score;
      //bid_t记录为要选择的块
      for (int i = 0;i < pre_cached.size();i++)
      {
        bid_t pre_choose = pre_cached[i];
        size_t score = 0;
        for (int j = 0;j < cached.size();j++)
        {
          score += walk_manager.block_walks[pre_choose * nblocks + cached[j]].block_numwalks() + walk_manager.block_walks[cached[j] * nblocks + pre_choose].block_numwalks();
        }
        score += walk_manager.block_walks[pre_choose * nblocks + pre_choose].block_numwalks();
        block_score.push_back(std::make_pair(pre_choose, score));
      }
      std::sort(block_score.begin(), block_score.end(), [](const std::pair<bid_t, size_t>& a, const std::pair<bid_t, size_t>& b) { return a.second > b.second; });
      if (block_score[0].second > 0 && block_score[0].second > conf.zero_threshold)
      {
        buckets.push_back(block_score[0].first);
        _m.stop_time("walkaware_choose_blocks");
        return true;
      }
    }

    _m.stop_time("walkaware_choose_blocks");
    return false;
  }

  bool cggraph_choose(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    static int k = 0;
    if (k > 0)
    {
      return true;
    }
    std::vector<std::pair<bid_t, float>> degs;
    for (int i = 0;i < walk_manager.nblocks;i++)
    {
      degs.push_back(std::make_pair(i, walk_manager.global_blocks->blocks[i].nedges / walk_manager.global_blocks->blocks[i].nverts));
    }
    std::sort(degs.begin(), degs.end(), [](const auto& a, const auto& b) {return a.second > b.second; });
    std::cout << "cggraph choose block :";
    for (int i = 0;i < cache.ncblock;i++)
    {
      buckets.push_back(degs[i].first);
      std::cout << "cggraph choose block " << degs[i].first << "  ";
    }
    std::cout << std::endl;
    k++;
    return true;
  }

  bool cpu_cggraph(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, metrics& _m)
  {
    cpu_blk.clear();
    for (int i = 0;i < walk_manager.nblocks;i++)
    {
      for (int j = 0;j < walk_manager.nblocks;j++)
      {
        bid_t totblk = i * walk_manager.nblocks + j;
        if (std::find(buckets.begin(), buckets.end(), i) != buckets.end() && std::find(buckets.begin(), buckets.end(), j) != buckets.end())
          continue; // 这两个块都在gpu上
        if (walk_manager.nblockwalks(totblk) > 0)
        {
          cpu_blk.push_back(totblk);
        }
      }
    }
    return true;
  }

  bool schedule(graph_config& conf, graph_cache& cache, graph_driver& driver, graph_walk& walk_manager, gpu_cache* g_cache, gpu_graph* g_graph, graph_cache& c_cache, cudaStream_t cstream, metrics& _m, cudaStream_t copy)
  {
    bool res = false;
    if (conf.walkaware == true)
    {
      res = walkaware(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
      cpu_firstcol(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
      return res;
    }
    if (conf.cggraph == true)
    {
      res = cggraph_choose(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
      cpu_cggraph(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
      return res;
    }
    if (conf.gpu_schedule == true)
    {
      _m.start_time("max_choose_blocks");
      res = max_choose_blocks(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
      _m.stop_time("max_choose_blocks");
    }
    else {
      res = basic_choose(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
    }
    if (conf.cpu_schedule == true)
    {
      cpu_firstcol(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
    }
    else
    {
      cpu_firstrow(conf, cache, driver, walk_manager, g_cache, g_graph, c_cache, _m);
    }
    return res;
  }
};

#endif
