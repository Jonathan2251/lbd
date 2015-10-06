// clang -target mips-unknown-linux-gnu -c ch4_1_3.cpp -emit-llvm -o ch4_1_3.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_3.bc -o -

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_3.bc -o -

/// start
int test_rotate_left()
{
  unsigned int a = 8;
  int result = ((a << 30) | (a >> 2));
  
  return result;
}

#ifdef TEST_ROXV

int test_rotate_left1(unsigned int a, int n)
{
  int result = ((a << n) | (a >> (32 - n)));
  
  return result;
}

int test_rotate_right(unsigned int a, int n)
{
  int result = ((a >> n) | (a << (32 - n)));
  
  return result;
}

#endif