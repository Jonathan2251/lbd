#include <iostream>

template <typename T>
void f(T x) {
    std::cout << "Template f(T)" << std::endl;
}

void f(int x) {
    std::cout << "Non-template f(int)" << std::endl;
}

int main() {
    f(42);        // Which one gets called?
    f('a');       // Template or non-template?
    f<int>('a');  // Explicit template instantiation
}

