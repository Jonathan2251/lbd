// clang -target mips-unknown-linux-gnu -c ch4_2.cpp -emit-llvm -o ch4_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_2.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -view-isel-dags -view-sched-dags -relocation-model=pic -filetype=asm ch4_2.bc -o -

/// start
int test_mod()
{
  int b = 11;
//  unsigned int b = 11;
  
  b = (b+1)%12;
  
  return b;
}
