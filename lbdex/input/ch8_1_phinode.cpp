// clang -O1 -target mips-unknown-linux-gnu -c ch8_1_phinode.cpp -emit-llvm -o ch8_1_phinode.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch8_1_phinode.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/opt -O1 -S ch8_1_phinode.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_1_phinode.bc -o -

/// start
int test_phinode(int a , int b)
{
  int c = 2;
  
  if (a == 0) {
    a++; // a = 1
  }
  else if (b != 0) {
    a--; // b = 2
  }
  c = a + b;
  
  return c;
}
