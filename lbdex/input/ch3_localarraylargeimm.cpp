// clang -target mips-unknown-linux-gnu -c ch3_localarraylargeimm.cpp -emit-llvm -o ch3_localarraylargeimm.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3_localarraylargeimm.bc -o -

/// start
int main()
{
  int A[100000];
  A[99990] = 6;
  int a = 5;
  int b = 2;

  int c = a + b;      // c = 7
  int d = b + 1;      // d = 3

  return (c+d);
}

