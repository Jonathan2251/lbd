// clang -target mips-unknown-linux-gnu -c ch4_2_logic.cpp -emit-llvm -o ch4_2_logic.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch4_2_logic.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch4_2_logic.bc -o -


/// start
int test_andorxornot()
{
  int a = 5;
  int b = 3;
  int c = 0, d = 0, e = 0;

  c = (a & b);  // c = 1
  d = (a | b);  // d = 7
  e = (a ^ b);  // e = 6
  b = !a;       // b = 0
  
  return (c+d+e+b); // 14
}

int test_setxx()
{
  int a = 5;
  int b = 3;
  int c, d, e, f, g, h;
  
  c = (a == b); // seq, c = 0
  d = (a != b); // sne, d = 1
  e = (a < b);  // slt, e = 0
  f = (a <= b); // sle, f = 0
  g = (a > b);  // sgt, g = 1
  h = (a >= b); // sge, g = 1
  
  return (c+d+e+f+g+h); // 3
}

