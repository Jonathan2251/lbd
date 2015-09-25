// clang -target mips-unknown-linux-gnu -c ch7_5-global-array.cpp -emit-llvm -o ch7_5-global-array.bc
// /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llvm-dis ch7_5-global-array.bc -o ch7_5-global-array.ll

/// start
int a[3] = {2012, 10, 12};

int test_struct()
{
  int i = a[1];

  return i; // 10
}

