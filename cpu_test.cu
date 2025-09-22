#include"api/threadpool.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <stdio.h>
#include<stdint.h>
#include <cstdlib>  // 包含 rand()、srand()
#include <ctime>    // 包含 time()，用于设置种子
#include <cstdlib>
#include <memory.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/time.h>
#include <vector>
#include <cassert>
#include <list>
#include <iostream>
#include"api/types.hpp"
void compute(void* arg) {
    int **data= (int**)arg;
    for(int i=0;i<1000000;i++)
    {
        for(int j=0;j<1000;j++)
        {
            int rand= std::rand() % 1000;
            int d=data[j%rand][rand];
            d*=2;
            data[j%rand][rand]=d;
            d=data[j%rand][rand];
            d/=2;
            data[j%rand][rand]=d;
        }
    }
}
int main() {
    ThreadPool pool(64, 512);

    int **data=(int**)malloc(1000*sizeof(int*));
    for(int i=0;i<1000;i++)
    {
        data[i]=(int*)malloc(1000*sizeof(int));
        for(int j=0;j<1000;j++)
        {
            data[i][j]=i*j;
        }
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        pool.pushJob(compute, data, sizeof(data));
    }

    // 等待所有任务完成
    pool.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "All tasks completed in " << elapsed_seconds.count() << " seconds.\n";
    // 清理示例数据
    for (int i = 0; i < 1000; ++i) {
        delete[] data[i];
    }
    delete[] data;

    return 0;
}