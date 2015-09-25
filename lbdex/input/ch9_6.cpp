// clang -target mips-unknown-linux-gnu -c ch9_6.cpp -emit-llvm -o ch9_6.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_6.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_6.bc -o -

/// start

int sum_i(int x1, int x2, int x3, int x4, int x5, int x6)
{
  int sum = x1 + x2 + x3 + x4 + x5 + x6;
  
  return sum; 
}

int sum_f(int x1, int x2, int x3)
{
  int sum = (int)(x1 + x2 + x3);
  
  return sum; 
}

int sum_d(long long x1, long long x2)
{
  int sum = (int)(x1 + x2);
  
  return sum; 
}

int sum_d2(double x1, double x2)
{
  int sum = (int)(x1 + x2);
  
  return sum; 
}

int test_sum_i_f()
{
  int a = 0;
  float b[6] = {1.1, 2.1, 3.1};
  double c[6] = {4.4, 5.7};
  a = sum_i(1, 2, 3, 4, 5, 6); // 15
  a += sum_f((int)b[0], (int)b[1], (int)b[2]); // 15+6=21
  a += sum_d((long long)c[0], (long long)c[1]); // 21+10=31
  a += sum_d2((long long)c[0], (long long)c[1]); // 31+10=41
  
  return a;
}
