#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <cassert>

std::vector<int> data;
std::atomic<bool> ready_flag(false);

void producer_thread() {
    // Populate data
    for (int i = 0; i < 10; ++i) {
        data.push_back(i * 10);
    }
    // Release store: Ensures all writes to 'data' are visible before 'ready_flag' is set.
    ready_flag.store(true, std::memory_order_release); 
}

void consumer_thread() {
    // Acquire load: Ensures all writes that happened before the release store are visible.
    while (!ready_flag.load(std::memory_order_acquire)) {
        // Spin-wait until data is ready
    }
    // Now 'data' is guaranteed to be fully populated and visible
    for (int i = 0; i < 10; ++i) {
        assert(data[i] == i * 10);
    }
    std::cout << "Consumer successfully read data." << std::endl;
}

int main() {
    std::thread producer(producer_thread);
    std::thread consumer(consumer_thread);

    producer.join();
    consumer.join();

    return 0;
}

