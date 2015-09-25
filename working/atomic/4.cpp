// clang++ -pthread -std=c++11 -S 4.cpp -emit-llvm -o 4.ll
// Generate IRs "fence"

#include <iostream>       // std::cout
#include <atomic>         // std::atomic
#include <thread>         // std::thread

int a;

void func1()
{
    for(int i = 0; i < 1000000; ++i)
    {
        a = i;
        // Ensure that changes to a to this point are visible to other threads
        atomic_thread_fence(std::memory_order_release);
    }
}

void func2()
{
    for(int i = 0; i < 1000000; ++i)
    {
        // Ensure that this thread's view of a is up to date
        atomic_thread_fence(std::memory_order_acquire);
        std::cout << a;
    }
}

int main()
{
    std::thread t1 (func1);
    std::thread t2 (func2);

    t1.join(); t2.join();
}

