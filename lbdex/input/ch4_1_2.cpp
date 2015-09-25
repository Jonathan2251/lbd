// clang -target mips-unknown-linux-gnu -c ch4_1_2.cpp -emit-llvm -o ch4_1_2.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm -cpu0-enable-overflow=true ch4_1_2.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj -cpu0-enable-overflow=true ch4_1_2.bc -o ch4_1_2.cpu0.o

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm -cpu0-enable-overflow=true ch4_1_2.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj -cpu0-enable-overflow=true ch4_1_2.bc -o ch4_1_2.cpu0.o

/// start
#include "debug.h"

int test_add_overflow()
{
  int a = 0x70000000;
  int b = 0x20000000;
  int c = 0;

  c = a + b;
  
  return 0;
}

int test_sub_overflow()
{
  int a = -0x70000000;
  int b = 0x20000000;
  int c = 0;

  c = a - b;
  
  return 0;
}
