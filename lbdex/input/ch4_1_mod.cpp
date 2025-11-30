// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch4_1_mod.cpp  -emit-llvm -o ch4_1_mod.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_mod.bc -o -

/// start
int test_mod()
{
  int b = 11;
  volatile int a = 12;
  
  b = (b+1)%a;
  
  return b;
}
