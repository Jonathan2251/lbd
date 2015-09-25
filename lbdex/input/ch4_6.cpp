// clang -target mips-unknown-linux-gnu -c ch4_6.cpp -emit-llvm -o ch4_6.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch4_6.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=static -filetype=asm ch4_6.bc -o -


/// start
int test_OptSlt()
{
  int a = 3, b = 1;
  int d = 0, e = 0, f = 0;

  d = (a < 1);
  e = (b < 2);
  f = d + e;

  return (f);
}
