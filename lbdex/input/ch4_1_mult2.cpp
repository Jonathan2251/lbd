// clang -target mips-unknown-linux-gnu -c ch4_1_mult2.cpp -emit-llvm -o ch4_1_mult2.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_mult2.bc -o -

/// start
int test_mult()
{
  int b = 11;
  int a = 12;

  b = (b+1)%a;
  
  return b;
}
