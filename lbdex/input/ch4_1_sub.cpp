// clang -target mips-unknown-linux-gnu -c ch4_1_sub.cpp -emit-llvm -o ch4_1_sub.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_sub.bc -o -

#include <stdio.h>

int test_sub()
{
  int a = 0x7FFFFFFF;
  int b = 0x80000001;

  int d = a - b;      // d = 3
  return d;
}

int main() {
  int res = test_sub();
  printf("res: %d\n", res);
}
