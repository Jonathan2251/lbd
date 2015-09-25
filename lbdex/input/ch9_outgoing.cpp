// clang -O3 -target mips-unknown-linux-gnu -c ch9_outgoing.cpp -emit-llvm -o ch9_outgoing.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch9_outgoing.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -view-dag-combine1-dags -relocation-model=static -filetype=asm ch9_outgoing.bc -o -

/// start
extern int sum_i(int x1);

int call_sum_i() {
  return sum_i(1);
}
