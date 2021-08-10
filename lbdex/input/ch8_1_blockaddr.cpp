// clang -target mips-unknown-linux-gnu -c ch8_1_blockaddr.cpp -emit-llvm -o ch8_1_blockaddr.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch8_1_blockaddr.bc -o -

/// start
int test_blockaddress(int x) {
  const void *addr = &&FOO;
  if (x == 1)
    goto *addr;

  return 2;
FOO:
  return 1;
}
