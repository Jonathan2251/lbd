// clang -target mips-unknown-linux-gnu -c ch4_1_1.cpp -emit-llvm -o ch4_1_1.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_1.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1_1.bc -o ch4_1_2.cpu0.o

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1_1.bc -o ch4_1_1.cpu0.o

/// start
int test_math()
{
  int a = 5;
  int b = 2;
  unsigned int a1 = -5;
  int c, d, e, f, g, h, i;
  unsigned int f1, g1, h1, i1;

  c = a + b;      // c = 7
  d = a - b;      // d = 3
  e = a * b;      // e = 10
  f = (a << 2);   // f = 20
  f1 = (a1 << 1); // f1 = 0xfffffff6 = -10
  g = (a >> 2);   // g = 1
  g1 = (a1 >> 30); // g1 = 0x03 = 3
  h = (1 << a);   // h = 0x20 = 32
  h1 = (1 << b);  // h1 = 0x04
  i = (0x80 >> a); // i = 0x04
  i1 = (b >> a);  // i1 = 0x0

  return (c+d+e+f+int(f1)+g+(int)g1+h+(int)h1+i+(int)i1);
// 7+3+10+20-10+1+3+32+4+4+0 = 74
}

