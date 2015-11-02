// clang -O1 -target mips-unknown-linux-gnu -c ch8_2_select_global_pic.cpp -emit-llvm -o ch8_2_select_global_pic.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_2_select_global_pic.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_2_select_global_pic.bc -o -

// No need to verify. Build test only.

/// start
volatile int a1 = 1;
volatile int b1 = 2;

int gI1 = 100;
int gJ1 = 50;

int test_select_global_pic()
{
  if (a1 < b1)
    return gI1;
  else
    return gJ1;
}
