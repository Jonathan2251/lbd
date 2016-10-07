// clang -target mips-unknown-linux-gnu -c ch8_2_longbranch.cpp -emit-llvm -o ch8_2_longbranch.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm -force-cpu0-long-branch ch8_1_longbranch.bc -o -

/// start
int test_longbranch()
{
  volatile int a = 2;
  volatile int b = 1;
  int result = 0;

  if (a < b)
    result = 1;
  return result;
}

