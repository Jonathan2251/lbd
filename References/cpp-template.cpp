#include <iostream>

#if TEMPLATE==1
template <typename T>
void f(T x) {
    std::cout << "Template f(T)" << std::endl;
}
#endif

#if FUNCTION==1
void f(int x) {
    std::cout << "Non-template f(int)" << std::endl;
}
#endif

int main() {
#if (TEMPLATE==1) || (FUNCTION==1)
    f(42);        // Which one gets called?
    f('a');       // Template or non-template?
#endif
#if TEMPLATE==1
    f<int>('a');  // Explicit template instantiation
#endif
}

