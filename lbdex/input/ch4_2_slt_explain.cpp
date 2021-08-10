// clang -target mips-unknown-linux-gnu -c ch4_2_slt_explain.cpp -emit-llvm -o ch4_2_slt_explain.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch4_2_slt_explain.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=static -filetype=asm ch4_2_slt_explain.bc -o -


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
