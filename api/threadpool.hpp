#ifndef TINYSERVER_THREADPOOL_HPP
#define TINYSERVER_THREADPOOL_HPP
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <list>
#include <pthread.h>
class ThreadPool {
private:
    struct NWORKER {
        pthread_t threadid;
        tid_t tid;
        std::atomic<bool> terminate;
        std::atomic<bool> isWorking;
        ThreadPool* pool;
    } *m_workers;

    struct NJOB {
        void (*func)(void* arg);
        void* user_data;
    };
public:
    ThreadPool(int numWorkers, int max_jobs);
    ~ThreadPool();
    int pushJob(void (*func)(void* data), void* arg, int len);
    bool wait();

private:
    bool _addJob(NJOB* job);
    static void* _run(void* arg);
    void _threadLoop(void* arg);

private:
    std::list<NJOB*> m_jobs_list;
    int m_max_jobs;
    int m_sum_thread;
    int m_free_thread;
    pthread_cond_t m_jobs_cond;
    pthread_mutex_t m_jobs_mutex;
};

ThreadPool::ThreadPool(int numWorkers, int max_jobs = 10) : m_sum_thread(numWorkers), m_free_thread(numWorkers), m_max_jobs(max_jobs) {
    if (numWorkers < 1 || max_jobs < 1) {
        perror("workers num error");
    }
    if (pthread_cond_init(&m_jobs_cond, NULL) != 0)
        perror("init m_jobs_cond fail\n");

    if (pthread_mutex_init(&m_jobs_mutex, NULL) != 0)
        perror("init m_jobs_mutex fail\n");

    m_workers = new NWORKER[numWorkers];
    if (!m_workers) {
        perror("create workers failed!\n");
    }
    for (int i = 0; i < numWorkers; ++i) {

        m_workers[i].pool = this;
        m_workers[i].terminate.store(false);
        m_workers[i].isWorking.store(false);
        int ret = pthread_create(&(m_workers[i].threadid), NULL, _run, &m_workers[i]);
        if (ret) {
            delete[] m_workers;
            perror("create worker fail\n");
        }
        if (pthread_detach(m_workers[i].threadid)) {
            delete[] m_workers;
            perror("detach worder fail\n");
        }
        m_workers[i].tid = i;
    }
}

ThreadPool::~ThreadPool() {
    for (int i = 0; i < m_sum_thread; i++) {
        m_workers[i].terminate.store(true);
    }
    pthread_mutex_lock(&m_jobs_mutex);
    pthread_cond_broadcast(&m_jobs_cond);
    pthread_mutex_unlock(&m_jobs_mutex);
    delete[] m_workers;
}

bool ThreadPool::_addJob(NJOB* job) {
    pthread_mutex_lock(&m_jobs_mutex);
    if (m_jobs_list.size() >= m_max_jobs) {
        pthread_mutex_unlock(&m_jobs_mutex);
        return false;
    }
    m_jobs_list.push_back(job);
    pthread_cond_signal(&m_jobs_cond);
    pthread_mutex_unlock(&m_jobs_mutex);
    return true;
}

int ThreadPool::pushJob(void (*func)(void*), void* arg, int len) {
    struct NJOB* job = (struct NJOB*)malloc(sizeof(struct NJOB));
    if (job == NULL) {
        perror("NJOB malloc error");
        return -2;
    }

    memset(job, 0, sizeof(struct NJOB));

    job->user_data = malloc(len);
    memcpy(job->user_data, arg, len);
    job->func = func;

    bool res = _addJob(job);
    if (res)
    {
        return 1;
    }
    else {
        free(job->user_data);
        free(job);
        return 0;
    }
}
bool ThreadPool::wait() {
    for (int i = 0;i < m_sum_thread;i++)
    {
        while (m_workers[i].isWorking || m_free_thread < m_sum_thread || m_jobs_list.size()>0);
    }
}

void* ThreadPool::_run(void* arg) {
    NWORKER* worker = (NWORKER*)arg;
    worker->pool->_threadLoop(arg);
}

void ThreadPool::_threadLoop(void* arg) {
    NWORKER* worker = (NWORKER*)arg;
    while (1) {
        pthread_mutex_lock(&m_jobs_mutex);
        while (m_jobs_list.size() == 0) {
            if (worker->terminate.load()) break;
            pthread_cond_wait(&m_jobs_cond, &m_jobs_mutex);
        }
        if (worker->terminate.load()) {
            pthread_mutex_unlock(&m_jobs_mutex);
            break;
        }
        __sync_fetch_and_sub(&m_free_thread, 1);
        worker->isWorking.store(true);
        struct NJOB* job = m_jobs_list.front();
        m_jobs_list.pop_front();

        pthread_mutex_unlock(&m_jobs_mutex);
        memcpy(job->user_data, &worker->tid, sizeof(tid_t));
        job->func(job->user_data);
        worker->isWorking.store(false);
        __sync_fetch_and_add(&m_free_thread, 1);
        free(job->user_data);
        free(job);
    }

    free(worker);
    pthread_exit(NULL);
}

#endif //TINYSERVER_THREADPOOL_HPP