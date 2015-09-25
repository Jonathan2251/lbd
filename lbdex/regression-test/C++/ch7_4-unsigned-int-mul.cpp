// clang -target mips-unknown-linux-gnu -c ch7_4-unsigned-int-mul.cpp -emit-llvm -o ch7_4-unsigned-int-mul.bc
// /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llvm-dis ch7_4-unsigned-int-mul.bc -o ch7_4-unsigned-int-mul.ll

/// start
long long test_longlong()
{
  unsigned int a1 = 0x3001000;
  unsigned int b1 = 0x2001000;
  
  long long f = (long long)a1 * (long long)b1; // f = 0x00060050,01000000

  return f; // f = 0x00060050,01000000
}

