// clang -O3 -target mips-unknown-linux-gnu -c ch9_incoming.cpp -emit-llvm -o ch9_incoming.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch9_incoming.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -view-dag-combine1-dags -relocation-model=pic -filetype=asm ch9_incoming.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -view-dag-combine1-dags -relocation-model=pic -filetype=asm ch9_incoming.bc -o -

/// start
int sum_i(int x1, int x2, int x3)
{
  int sum = x1 + x2 + x3;
  
  return sum; 
}
