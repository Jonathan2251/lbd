#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

std::atomic<int> data(0);
std::atomic<bool> ready(false);

void producer() {
    // Populate data
    for (int i = 0; i < 10; ++i) {
        data.push_back(i * 10);
    }
    // Release store: Ensures all writes to 'data' are visible before 'ready' is set.
    ready.store(true, std::memory_order_release); // Release signal
}

void consumer() {
    // Acquire load: Ensures all writes that happened before the release store are visible.
    while (!ready.load(std::memory_order_acquire)); // Wait for ready signal

    // Ensure that this print sees the correct value of `data`
    for (int i = 0; i < 10; ++i) {
        std::cout << data[i];
    }
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);

    t1.join();
    t2.join();

    return 0;
}

