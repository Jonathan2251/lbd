#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> data(0);
std::atomic<bool> ready(false);

void producer() {
    data.store(42, std::memory_order_relaxed); // Store value
    ready.store(true, std::memory_order_release); // Release signal
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)); // Wait for ready signal

    // Ensure that this print sees the correct value of `data`
    std::cout << "Data: " << data.load(std::memory_order_relaxed) << std::endl;
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);

    t1.join();
    t2.join();

    return 0;
}

