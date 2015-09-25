// clang -target mips-unknown-linux-gnu -c ch9_1_3.cpp -emit-llvm -o ch9_1_3.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_1_3.bc -o -

/// start
extern int sum_i(int x, int y);

int main()
{
  int b = 1;
  int c = 2;
  int a = sum_i(b, c);
  
  return a;
}
