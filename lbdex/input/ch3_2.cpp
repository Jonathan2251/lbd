// clang -target mips-unknown-linux-gnu -c ch3_2.cpp -emit-llvm -o ch3_2.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3_2.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3_2.bc -o -

/// start
int main()
{
  int a = 5;
  int b = 2;

  int c = a + b;      // c = 7
  int d = b + 1;      // d = 3

  return (c+d);
}

