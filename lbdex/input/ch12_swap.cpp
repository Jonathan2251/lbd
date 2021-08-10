// clang++ -c ch12_swap.cpp -emit-llvm -o ch12_swap.bc
// ~/llvm/test/build/bin/llvm-dis ch12_swap.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch12_swap.bc -o -

#include <iostream>
using namespace std;

int main()
{
  int a = 1;
  int b = 2;

  swap(a, b);

  return a;
}
