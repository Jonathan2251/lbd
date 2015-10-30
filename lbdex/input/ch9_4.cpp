// clang -c ch9_4.cpp -emit-llvm -o ch9_4.bc // for #include <stdlib.h>
// clang -target mips-unknown-linux-gnu -c ch9_4.cpp -emit-llvm -o ch9_4.bc

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false -relocation-model=pic -filetype=asm ch9_4.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false -relocation-model=pic -filetype=asm ch9_4.bc -o -

/// start

// This file needed compile without option, -target mips-unknown-linux-gnu, so 
// it is verified by build-run_backend2.sh or verified in lld linker support
// (build-slinker.sh).

//#include <alloca.h>
//#include <stdlib.h>

int sum(int x1, int x2, int x3, int x4, int x5, int x6)
{
  int sum = x1 + x2 + x3 + x4 + x5 + x6;
  
  return sum; 
}

int weight_sum(int x1, int x2, int x3, int x4, int x5, int x6)
{
//  int *b = (int*)alloca(sizeof(int) * 1 * x1);
  int* b = (int*)__builtin_alloca(sizeof(int) * 1 * x1);
  int *a = b;
  *b = x3;

  int weight = sum(3*x1, x2, x3, x4, 2*x5, x6);

  return (weight + (*a));
}

int test_alloc()
{
//  ENABLE_TRACE;
  int a = weight_sum(1, 2, 3, 4, 5, 6); // 31
  
  return a;
}
