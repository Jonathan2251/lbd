// clang -O0 -c ch8_3.cpp -emit-llvm -o ch8_3.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch8_3.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch8_3.bc -o -

/// start

// The following files will generate IR select even compile with clang -O0.
int test_movx_1()
{
  volatile int a = 1;
  int c = 0;

  c = !a ? 1:3;

  return c;
}

int test_movx_2()
{
  volatile int a = 1;
  int c = 0;

  c = a ? 1:3;

  return c;
}

