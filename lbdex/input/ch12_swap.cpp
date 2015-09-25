// clang++ -c ch14_swap.cpp -emit-llvm -o ch14_swap.bc
// ~/llvm/test/cmake_debug_build/bin/llvm-dis ch14_swap.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch14_swap.bc -o -

#include <iostream>
using namespace std;

int main()
{
  int a = 1;
  int b = 2;

  swap(a, b);

  return a;
}
