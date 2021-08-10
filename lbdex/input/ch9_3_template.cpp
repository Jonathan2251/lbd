// clang -target mips-unknown-linux-gnu -c ch9_3_template.cpp -emit-llvm -o ch9_3_template.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_template.bc -o ch9_3_template.cpu0.s
// ~/llvm/test/build/bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_3_template.bc -o ch9_3_template.mips.s

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
