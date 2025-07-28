#ifndef _GRAPH_BUFFER_H_
#define _GRAPH_BUFFER_H_

#include <assert.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include"types.hpp"
/** This file defines the buffer data structure used in graph processing */

template <typename T>
class graph_buffer
{
public:
    size_t bsize; // buffer current size
    size_t capacity;
    T* array;
    graph_buffer<T>* next;
    graph_buffer() { this->bsize = this->capacity = 0, this->array = NULL, this->next = NULL; }
    graph_buffer(size_t size)
    {
        alloc(size);
    }
    ~graph_buffer() { this->destroy(); }

    void alloc(size_t size)
    {
        this->capacity = size;
        this->bsize = 0;
        checkCudaError(cudaHostAlloc((void**)&(this->array), size * sizeof(T), cudaHostAllocMapped));
        this->next = NULL;
    }
    bool alloc()
    {
        if (capacity <= 0)
            return false;
        this->bsize = 0;
        checkCudaError(cudaHostAlloc((void**)&(this->array), capacity * sizeof(T), cudaHostAllocMapped));
        this->next = NULL;
    }

    void realloc(size_t size)
    {
        this->array = (T*)malloc(this->array, size * sizeof(T));
        this->capacity = size;
        this->bsize = 0;
        this->next = NULL;
    }

    void destroy()
    {
        if (this->array)
            cudaFreeHost(this->array);
        if (this->next)
            next->~graph_buffer();
        this->array = NULL;
        this->bsize = 0;
        this->capacity = 0;
        this->next = NULL;
    }

    T& operator[](size_t off)
    {
        graph_buffer<T>* p = this;
        while (off >= p->bsize)
        {
            off -= p->bsize;
            p = p->next;
        }
        return p->array[off];
    }

    T*& buffer_begin() { return this->array; }
    size_t size()
    {
        graph_buffer<T>* p = this;
        size_t size = 0;
        while (p != NULL)
        {
            size += p->bsize;
            p = p->next;
        }
        return size;
    }

    bool push_back(T val)
    {
        graph_buffer<T>* p = this;
        while (p != NULL)
        {
            if (p->full())
            {
                if (p->next != NULL)
                    p = p->next;
                else
                {
                    p->next = (graph_buffer<T> *)malloc(sizeof(graph_buffer<T>));
                    p->next->alloc(p->capacity);
                    p->next->push_back(val);
                    return true;
                }
            }
            else
            {
                if (p->array == NULL)
                    p->alloc(p->capacity);
                p->array[p->bsize] = val;
                p->bsize++;
                return true;
            }
        }
        return false;
    }
    int push(vid_t pv, vid_t cv, bid_t pb, bid_t cb, hid_t hop)
    {
        if (bsize >= capacity)
        {
            return 0;
        }
        if (array == NULL)
        {
            return 0;
        }
        array[bsize].previous = pv;
        array[bsize].current = cv;
        array[bsize].prev_index = pb;
        array[bsize].cur_index = cb;
        array[bsize].hop = hop;
        bsize++;
        if (bsize >= capacity)
        {
            return 2;
        }
        return 1;
    }

    T load(wid_t w)
    {
        graph_buffer<T>* p = this;
        while (w >= p->bsize)
        {
            p = p->next;
            w = w - p->bsize;
        }
        p->bsize--;
        T walk = p->array[p->bsize];
        p = this;
        while (p->next != NULL)
        {
            if (p->next->bsize == 0)
                p->next->destroy();
            p = p->next;
        }
        return walk;
    }

    bool empty() { return this->bsize == 0; }
    bool full() { return this->bsize == this->capacity; }
    void emptyarray()
    {
        this->array = NULL;
        this->bsize = 0;
        return;
    }

    /** test if add num elements whether will overflow the maximum capacity or not. */
    bool test_overflow(size_t num)
    {
        return this->bsize + num > this->capacity;
    }

    void clear()
    {
        this->bsize = 0;
        if (this->next)
        {
            this->next->destroy();
        }
    }

    void set_size(size_t _size)
    {
        this->bsize = _size;
    }
};

#endif