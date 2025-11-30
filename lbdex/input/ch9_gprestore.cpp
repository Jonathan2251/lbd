// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_gprestore.cpp -emit-llvm -o ch9_gprestore.bc
// ~/llvm/test/build/bin/llvm-dis ch9_gprestore.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch9_gprestore.bc -o -

// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch9_gprestore.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch9_gprestore.bc -cpu0-no-cpload -cpu0-reserve-gp -o -

/// start
extern int sum_i(int x1);

int call_sum_i() {
  int a = sum_i(1);
  a += sum_i(2);
  return a;
}
