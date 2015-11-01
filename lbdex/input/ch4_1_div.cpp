// clang -target mips-unknown-linux-gnu -c ch4_1_div.cpp -emit-llvm -o ch4_1_div.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_div.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1_div.bc -o ch4_1_div.cpu0.o

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_div.bc -o -

/// start
int test_div()
{
  int a = -5;
  int b = 2;
  int c = 0x1000000;
  int d = 0;
  unsigned int a1 = -5, d1 = 0;

  d = a / b;    // d = -2
  d1 = a1 / c;  // a1 = 0xfffffffb, d1 = 0xff = 255

  return (d+d1); // 253
}

