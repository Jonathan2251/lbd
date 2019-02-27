// ~/llvm/release/cmake_debug_build/Debug/bin/clang -O3 -target mips-unknown-linux-gnu -c ch8_2_phinode.cpp -emit-llvm -o ch8_2_phinode.bc
// ~/llvm/release/cmake_debug_build/Debug/bin/clang -O0 -target mips-unknown-linux-gnu -c ch8_2_phinode.cpp -emit-llvm -o ch8_2_phinode.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch8_2_phinode.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_2_phinode.bc -o -

/// start
int test_phinode(int a , int b, int c)
{
  int d = 2;
  
  if (a == 0) {
    a = a+1; // a = 1
  }
  else if (b != 0) {
    a = a-1;
  }
  else if (c == 0) {
    a = a+2;
  }
  d = a + b;
  
  return d;
}
