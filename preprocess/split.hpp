#ifndef _GRAPH_SPLIT_H_
#define _GRAPH_SPLIT_H_

#include <string>
#include <vector>
#include <fstream>
#include "api/graph_buffer.hpp"
#include "api/constants.hpp"
#include "api/types.hpp"
#include "logger/logger.hpp"
#include "util/util.hpp"
#include "util/io.hpp"

/* split the beg_pos into multiple blocks, each block max size is BLOCKSIZE */
size_t split_blocks(const std::string &base_name, int fnum, size_t block_size)
{
    size_t max_nedges = block_size / sizeof(vid_t);
    size_t max_nverts = block_size / sizeof(eid_t);
    logstream(LOG_INFO) << "start split blocks, blocksize = " << block_size / (1024 * 1024) << "MB, max_nedges = " << max_nedges<< " ,  max_nverts = "<<max_nverts<< std::endl;

    vid_t cur_pos = 0;
    eid_t rd_edges = 0;         /* read edges */
    vid_t rd_verts = 0;         /* read vertices */
    std::vector<vid_t> vblocks; /* vertex blocks */
    std::vector<eid_t> eblocks; /* edge   blocks */
    vblocks.push_back(cur_pos);
    eblocks.push_back(rd_edges);

    std::string name = get_beg_pos_name(base_name);
    int fd = open(name.c_str(), O_RDONLY);
    assert(fd >= 0);
    vid_t nvertices = lseek(fd, 0, SEEK_END) / sizeof(eid_t);
    logstream(LOG_INFO) << "split blocks, nvertics = " << nvertices - 1 << std::endl;
    eid_t *beg_pos = (eid_t *)malloc(VERT_SIZE * sizeof(eid_t));//分配最大64M个顶点
    assert(beg_pos != NULL);

    vid_t rv;
    vid_t prev_ver=0;
    while (rd_verts < nvertices)
    {
        rv = min_value(nvertices - rd_verts, VERT_SIZE);
        load_block_range(fd, beg_pos, (size_t)rv, (off_t)rd_verts * sizeof(eid_t));
        for (vid_t v = 0; v < rv; v++)
        {
            if (beg_pos[v] - rd_edges > max_nedges)
            {
                // logstream(LOG_INFO) << "Block " << vblocks.size() - 1 << " : [ " << cur_pos << ", " << rd_verts + v - 1 << " ), csr position : [ " << rd_edges << ", " << beg_pos[v - 1] << " )" <<std::endl;
                // logstream(LOG_INFO) << "Block size:" <<(beg_pos[v-1]-rd_edges)*sizeof(vid_t)/(1024*1024)<<"MB" <<std::endl;
                cur_pos = rd_verts + v - 1;
                vblocks.push_back(cur_pos);
                rd_edges = beg_pos[v - 1];
                eblocks.push_back(rd_edges);
            }
            else if((rd_verts + v )-cur_pos >= max_nverts)
            {
                logstream(LOG_INFO) << "Block " << vblocks.size() - 1 << " : [ " << cur_pos << ", " << rd_verts + v << " ), csr position : [ " << rd_edges << ", " << beg_pos[v] << " )" <<std::endl;
                logstream(LOG_INFO) << "Block size:" <<(rd_verts + v-cur_pos+1)*sizeof(eid_t)/(1024*1024)<<"MB" <<std::endl;
                cur_pos= rd_verts + v;
                vblocks.push_back(cur_pos);
                rd_edges = beg_pos[v];
                eblocks.push_back(rd_edges);
            }
        }
        rd_verts += rv;
    }

    logstream(LOG_INFO) << "Block " << vblocks.size() - 1 << " : [ " << cur_pos << ", " << nvertices - 1 << " ), csr position : [ " << rd_edges << ", " << beg_pos[rv - 1] << " )" << std::endl;
    logstream(LOG_INFO) << "Total blocks num : " << vblocks.size() << std::endl;
    close(fd);
    vblocks.push_back(nvertices - 1);
    rd_edges = beg_pos[rv - 1];
    eblocks.push_back(rd_edges);

    /** write the vertex split points into vertex block file */
    std::string vblockfile = get_vert_blocks_name(base_name, block_size);
    auto vblf = std::fstream(vblockfile.c_str(), std::ios::out | std::ios::binary);
    vblf.write((char *)&vblocks[0], vblocks.size() * sizeof(vid_t));
    vblf.close();

    /** write the edge split points into edge block file */
    std::string eblockfile = get_edge_blocks_name(base_name, block_size);
    auto eblf = std::fstream(eblockfile.c_str(), std::ios::out | std::ios::binary);
    eblf.write((char *)&eblocks[0], eblocks.size() * sizeof(eid_t));
    eblf.close();

    return vblocks.size() - 1;
}

size_t split_blocks_metis(const std::string &base_name, const std::string &partset, int fnum)
{
    logstream(LOG_INFO) << "start split blocks metis " << std::endl;
    std::vector<vid_t> vblocks; /* vertex blocks */
    std::vector<eid_t> eblocks; /* edge   blocks */
    std::ifstream part(partset);
    std::string s;
    eid_t prefix = 0;
    eid_t max = 0;
    bid_t blk = 0;
    vid_t start_ver = 0;
    vid_t nver = 0;
    eid_t nedge = 0;
    std::string name = get_beg_pos_name(base_name);
    int fd = open(name.c_str(), O_RDONLY);
    assert(fd >= 0);
    vid_t nvertices = lseek(fd, 0, SEEK_END) / sizeof(eid_t);
    logstream(LOG_INFO) << "split blocks, nvertics = " << nvertices - 1 << std::endl;
    while (getline(part, s))
    {
        std::istringstream iss(s);
        iss >> blk;
        iss >> start_ver;
        iss >> nver;
        iss >> nedge;
        if (max < nedge)
            max = nedge;
        vblocks.push_back(start_ver);
        eblocks.push_back(prefix);
        prefix += nedge;
    }
    size_t blocksize = max * sizeof(vid_t);
    eblocks.push_back(prefix);
    vblocks.push_back(nvertices - 1);
    close(fd);
    /** write the vertex split points into vertex block file */
    std::string vblockfile = get_vert_blocks_name(base_name, blocksize);
    auto vblf = std::fstream(vblockfile.c_str(), std::ios::out | std::ios::binary);
    vblf.write((char *)&vblocks[0], vblocks.size() * sizeof(vid_t));
    vblf.close();

    /** write the edge split points into edge block file */
    std::string eblockfile = get_edge_blocks_name(base_name, blocksize);
    auto eblf = std::fstream(eblockfile.c_str(), std::ios::out | std::ios::binary);
    eblf.write((char *)&eblocks[0], eblocks.size() * sizeof(eid_t));
    eblf.close();
}
#endif
