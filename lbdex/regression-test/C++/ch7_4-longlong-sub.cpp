// clang -target mips-unknown-linux-gnu -c ch7_4-longlong-sub.cpp -emit-llvm -o ch7_4-longlong-sub.bc
// /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llvm-dis ch7_4-longlong-sub.bc -o ch7_4-longlong-sub.ll

/// start
long long test_longlong()
{
  long long a = 0x300000002;
  long long b = 0x100000001;
  
  long long c = a - b;   // c = 0x00000004,00000003

  return c; // c = 0x00000002,00000001
}

