// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_1.cpp -emit-llvm -o ch9_1.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_1.bc -o -
// ~/llvm/test/build/bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_1.bc -o -

/// start
int gI = 100;

int sum_i(int x1, int x2, int x3, int x4, int x5, int x6)
{
  int sum = gI + x1 + x2 + x3 + x4 + x5 + x6;
  
  return sum; 
}

int main()
{ 
  int a = sum_i(1, 2, 3, 4, 5, 6);  
  
  return a;
}
