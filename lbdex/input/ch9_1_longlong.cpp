// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_1_longlong.cpp -emit-llvm -o ch9_1_longlong.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch9_1_longlong.bc -o -
// ~/llvm/test/build/bin/llc -march=mips -relocation-model=static -filetype=asm ch9_1_longlong.bc -o -

/// start

long long sum_longlong(long long x1, long long x2, long long x3, long long x4, long long x5, long long x6)
{
  long long sum = x1 + x2 + x3 + x4 + x5 + x6;
  
  return sum; 
}

int test_sum_longlong()
{ 
  long long a = sum_longlong(1, 2, 3, 4, 5, -6);  
  
  return (int)a; // 9
}
