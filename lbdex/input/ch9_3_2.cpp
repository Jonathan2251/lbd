// clang -target mips-unknown-linux-gnu -c ch9_3_2.cpp -emit-llvm -o ch9_3_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_2.bc -o ch9_3_2.cpu0.s
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_3_2.bc -o ch9_3_2.mips.s

/// start
#include <stdarg.h>

template<class T>
T sum(T amount, ...)
{
  T i = 0;
  T val = 0;
  T sum = 0;
	
  va_list vl;
  va_start(vl, amount);
  for (i = 0; i < amount; i++)
  {
    val = va_arg(vl, T);
    sum += val;
  }
  va_end(vl);
  
  return sum; 
}

int test_template()
{
  int a = sum<int>(6, 0, 1, 2, 3, 4, 5);
	
  return a;
}
