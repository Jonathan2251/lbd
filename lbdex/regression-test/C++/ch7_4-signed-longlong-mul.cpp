// clang -target mips-unknown-linux-gnu -c ch7_4-signed-longlong-mul.cpp -emit-llvm -o ch7_4-signed-longlong-mul.bc
// /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llvm-dis ch7_4-signed-longlong-mul.bc -o ch7_4-signed-longlong-mul.ll

/// start
long long test_longlong()
{
  long long a = 0x300000002;
  long long b = 0x100000001;
  
  long long e = a * b;   // e = 0x00000005,00000002

  return e; // e = 0x00000005,00000002
}

