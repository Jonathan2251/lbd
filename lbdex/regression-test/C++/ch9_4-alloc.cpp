// Don't use -target mips-unknown-linux-gnu option in clang since my PC is 64 bits computer.

// clang -S ch9_4-alloc.cpp -emit-llvm -O3 -o ch9_4-alloc.ll

/// start
//#include <alloca.h>
#include <stdlib.h>

int sum(int x1, int x2, int x3, int x4, int x5, int x6)
{
  int sum = x1 + x2 + x3 + x4 + x5 + x6;
  
  return sum; 
}

int weight_sum(int x1, int x2, int x3, int x4, int x5, int x6)
{
  int *b = (int*)alloca(sizeof(int) * x1);
  *b = 1111;
  int weight = sum(6*x1, x2, x3, x4, 2*x5, x6);
  
  return weight; 
}

