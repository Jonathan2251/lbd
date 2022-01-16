// clang -target mips-unknown-linux-gnu -c ch9_3_vararg.cpp -emit-llvm -o ch9_3_vararg.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_vararg.bc -o -
// ~/llvm/test/build/bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_3_vararg.bc -o -

/// start
#include <stdarg.h>

int sum_i(int amount, ...)
{
  int i = 0;
  int val = 0;
  int sum = 0;
	
  va_list vl;
  va_start(vl, amount);
  for (i = 0; i < amount; i++)
  {
    val = va_arg(vl, int);
    sum += val;
  }
  va_end(vl);
  
  return sum; 
}

long long sum_ll(long long amount, ...)
{
  long long i = 0;
  long long val = 0;
  long long sum = 0;
	
  va_list vl;
  va_start(vl, amount);
  for (i = 0; i < amount; i++)
  {
    val = va_arg(vl, long long);
    sum += val;
  }
  va_end(vl);
  
  return sum; 
}

int test_vararg()
{
  int a = sum_i(6, 0, 1, 2, 3, 4, 5);
  long long b = sum_ll(6LL, 0LL, 1LL, 2LL, 3LL, -4LL, -5LL);
	
  return a+(int)b; // 13
}
