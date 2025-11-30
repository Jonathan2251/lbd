// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_3_bswap.cpp -emit-llvm -o ch9_3_bswap.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch9_3_bswap.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch9_3_bswap.bc -o -


/// start
int test_bswap16() {
  volatile int a = 0x1234;
  int result = (__builtin_bswap16(a) ^ 0x3412);
  
  return result;
}

int test_bswap32() {
  volatile int a = 0x1234;
  int result = (__builtin_bswap32(a) ^ 0x34120000);
  
  return result;
}

int test_bswap64() {
  volatile int a = 0x1234;
  int result = (__builtin_bswap64(a) ^ 0x3412000000000000);
  
  return result;
}

int test_bswap() {
  int result = test_bswap16() + test_bswap32() + test_bswap64();
  
  return result;
}
