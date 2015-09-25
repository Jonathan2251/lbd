// clang++ -pthread -std=c++11 -S 1.cpp -emit-llvm -o 1.ll
// Generate IRs "load atomic", "store atomic" and "atomicrmw"

#include <iostream>       // std::cout
#include <atomic>         // std::atomic
#include <thread>         // std::thread
#include <vector>         // std::vector

std::atomic<bool> ready (false);
std::atomic<bool> winner (false);

void count1m (int id) {
      while (!ready) {}                  // wait for the ready signal
        for (int i=0; i<1000000; ++i) {}   // go!, count to 1 million
          if (!winner.exchange(true)) { std::cout << "thread #" << id << " won!\n"; }
};

int main ()
{
      std::vector<std::thread> threads;
        std::cout << "spawning 10 threads that count to 1 million...\n";
          for (int i=1; i<=10; ++i) threads.push_back(std::thread(count1m,i));
            ready = true;
              for (auto& th : threads) th.join();

                return 0;
}
