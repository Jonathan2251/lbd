// clang -target mips-unknown-linux-gnu -c ch3_largeframe.cpp -emit-llvm -o ch3_largeframe.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3_largeframe.bc -o -

/// start
int test_largegframe() {
  int a[469753856];

  return 0;
}
