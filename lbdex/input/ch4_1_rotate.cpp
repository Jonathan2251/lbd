// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch4_1_3.cpp -emit-llvm -o ch4_1_3.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_3.bc -o -

/// start

//#define TEST_ROXV

int test_rotate_left()
{
  unsigned int a = 8;
  int result = ((a << 30) | (a >> 2));
  
  return result;
}

#ifdef TEST_ROXV

int test_rotate_left1()
{
  volatile unsigned int a = 4;
  volatile int n = 30;
  int result = ((a << n) | (a >> (32 - n)));
  
  return result;
}

int test_rotate_right()
{
  volatile unsigned int a = 1;
  volatile int n = 30;
  int result = ((a >> n) | (a << (32 - n)));
  
  return result;
}

#endif
